#include "BinFile.h"

#include <algorithm>
#include <cstring>
#include <iostream>

#include "miniz.h"

using namespace std::string_literals;

void BinFile::open(const std::string& filename) {
  close();
  fileStream_.open(filename, std::ios::in | std::ios::binary);
  if (!fileStream_) throw std::runtime_error("Can't open: "s + strerror(errno));
  try {
    fileRead(fileHeader.fileIdentifier);
    fileRead(fileHeader.fileFormatRev);
    if (std::string_view(&fileHeader.fileIdentifier[0], 11) != "DIRSIGPROTO")
      throw std::runtime_error("Bad file identifier, expected 'DIRSIGPROTO'");

    // Read byte order, set the swap endian flag accordingly.
    fileRead(fileHeader.byteOrder);
    if (fileHeader.byteOrder != 0 && fileHeader.byteOrder != 1) {
      std::cerr << int(fileHeader.byteOrder) << std::endl;
      throw std::runtime_error(
          "Bad byte order, expected 0 (big endian) or 1 (little endian)");
    }
    static uint32_t testU32 = 1;
    static uint8_t testU8 = *reinterpret_cast<uint8_t*>(&testU32);
    static bool bigEndian = testU8 == 0;
    swapEndian_ = ((fileHeader.byteOrder == 0) != bigEndian);

    // Read rest of header.
    fileReadMany(
        fileHeader.creationDateTime, fileHeader.dirsigVersion,
        fileHeader.simulationDesc, fileHeader.sceneOriginLatitude,
        fileHeader.sceneOriginLongitude, fileHeader.sceneOriginHeight,
        fileHeader.laserMountType, fileHeader.detectorMountType,
        fileHeader.xDetectorCount, fileHeader.yDetectorCount,
        fileHeader.xDetectorPitch, fileHeader.yDetectorPitch,
        fileHeader.xArrayOffset, fileHeader.yArrayOffset, fileHeader.lensK1,
        fileHeader.lensK2, fileHeader.taskCount, fileHeader.fpaId);
  } catch (const std::exception& error) {
    throw std::runtime_error("Can't read file header: "s + error.what());
  }

  // Scan through the file to cache task headers and pulse offsets.
  tasks.resize(fileHeader.taskCount);
  for (auto& task : tasks) {
    fileRead(task);
    task.pulseOffsets.resize(task.pulseCount);
    for (auto& pulseOffset : task.pulseOffsets) {
      pulseOffset = fileStream_.tellg();
      PulseHeader pulseHeader;
      fileRead(pulseHeader);
      fileStream_.seekg(pulseHeader.pulseDataBytes, std::ios::cur);
    }
  }
}

void BinFile::close() {
  fileStream_.close();
  tasks = {};
}

BinFile::PulseHeader BinFile::loadPulseHeader(
    const Task& task, size_t pulseIndex) try {
  if (pulseIndex >= task.pulseCount)
    throw std::runtime_error("Pulse index out of range");
  fileStream_.seekg(task.pulseOffsets[pulseIndex], std::ios::beg);
  PulseHeader pulseHeader;
  fileRead(pulseHeader);
  return pulseHeader;
} catch (const std::exception& error) {
  throw std::runtime_error("While loading pulse header: "s + error.what());
}

std::vector<double> BinFile::loadPulse(
    const Task& task, const PulseHeader& pulse) try {
  fileStream_.seekg(task.pulseOffsets[pulse.pulseIndex] + 913, std::ios::beg);
  std::vector<double> dataValues;
  dataValues.resize(
      fileHeader.xDetectorCount * fileHeader.yDetectorCount *
      (pulse.timeGateBinCount + 1) * pulse.samplesPerTimeBin);
  if (!pulse.dataCompression) {
    fileRead(dataValues.data(), dataValues.size(), 8);
  } else {
    std::vector<char> compressedData(pulse.pulseDataBytes);
    fileRead(compressedData.data(), compressedData.size(), 1);
    if (tinfl_decompress_mem_to_mem(
            dataValues.data(), dataValues.size() * sizeof(double),
            compressedData.data(), compressedData.size(),
            TINFL_FLAG_PARSE_ZLIB_HEADER) == TINFL_DECOMPRESS_MEM_TO_MEM_FAILED)
      throw std::runtime_error("Decompression failed!");
    if (swapEndian_)
      for (double& dataValue : dataValues)
        std::reverse(
            reinterpret_cast<char*>(&dataValue),
            reinterpret_cast<char*>(&dataValue) + sizeof(double));
  }
  return dataValues;
} catch (const std::exception& error) {
  throw std::runtime_error("While loading pulse: "s + error.what());
}

void BinFile::fileRead(void* ptr, size_t num, size_t sz) {
  if (!fileStream_.read(static_cast<char*>(ptr), num * sz))
    throw std::runtime_error("Read failed: "s + strerror(errno));
  if (swapEndian_ && sz > 1) {
    for (size_t i = 0; i < num; i++)
      std::reverse(
          static_cast<char*>(ptr) + (i + 0) * sz,
          static_cast<char*>(ptr) + (i + 1) * sz);
  }
}

void BinFile::fileRead(TaskHeader& header) try {
  fileReadMany(
      header.description, header.startDateTime, header.stopDateTime,
      header.focalLength, header.pulseRepetitionFreq, header.pulseDuration,
      header.pulseEnergy, header.laserSpectralCenter, header.laserSpectralWidth,
      header.pulseCount);
} catch (const std::exception& error) {
  throw std::runtime_error("While reading task header: "s + error.what());
}

void BinFile::fileRead(PulseHeader& header) try {
  fileReadMany(
      header.pulseTime, header.timeGateStart, header.timeGateStop,
      header.timeGateBinCount, header.samplesPerTimeBin,
      header.platformLocation, header.platformRotation,
      header.transmitterToMount, header.transmitterMountPointing,
      header.transmitterMountToPlatform, header.receiverToMount,
      header.receiverMountPointing, header.receiverMountToPlatform,
      header.pulseDataType, header.dataCompression, header.pulseIndex,
      header.pulseDataBytes, header.systemPolTransmit, header.systemPolReceive);
} catch (const std::exception& error) {
  throw std::runtime_error("While reading pulse header: "s + error.what());
}
