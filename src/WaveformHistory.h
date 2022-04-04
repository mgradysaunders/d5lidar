#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "VoxelGrid.h"

class WaveformHistory {
 public:
  static constexpr int32_t MaxWaveforms = 31;
  struct Record {
    int32_t waveforms[MaxWaveforms] = {};
    int32_t num = 0;

    void push(int32_t w) {
      if (num == MaxWaveforms)
        throw std::runtime_error("Max waveforms exceeded!");
      waveforms[num++] = w;
    }
  };

  void resize(const VoxelGrid& grid) {
    sizeX = grid.count[0];
    sizeY = grid.count[1];
    sizeZ = grid.count[2];
    records.clear();
    records.resize(sizeX * sizeY * sizeZ);
  }

  Record& operator[](Eigen::Vector3i index) noexcept {
    int32_t x = index[0];
    int32_t y = index[1];
    int32_t z = index[2];
    return records[x * sizeY * sizeZ + y * sizeZ + z];
  }

  int32_t sizeX = 0;
  int32_t sizeY = 0;
  int32_t sizeZ = 0;
  std::vector<Record> records;
};
