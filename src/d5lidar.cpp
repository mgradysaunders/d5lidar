#include <iomanip>

#include <Eigen/Geometry>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using EigenVectorView =
    Eigen::Map<Eigen::VectorXd, Eigen::Unaligned, Eigen::InnerStride<>>;

inline Eigen::Quaterniond rotateYZX(const double* xyz) {
  Eigen::Quaterniond rotateX, rotateY, rotateZ;
  rotateX = Eigen::AngleAxisd(xyz[0], Eigen::Vector3d(1, 0, 0));
  rotateY = Eigen::AngleAxisd(xyz[1], Eigen::Vector3d(0, 1, 0));
  rotateZ = Eigen::AngleAxisd(xyz[2], Eigen::Vector3d(0, 0, 1));
  return rotateX * rotateZ * rotateY;
}

namespace py = pybind11;
using namespace py::literals;

#include "BinFile.h"

constexpr double speedOfLight = 299792458;

template <size_t N>
static std::string charsToString(const char (&arr)[N]) {
  auto* end = &arr[0];
  while (end != &arr[0] + N and *end != '\0') end++;
  return std::string(&arr[0], end);
}

struct PrintVector3d {
  double x = 0;
  double y = 0;
  double z = 0;
  PrintVector3d() = default;
  PrintVector3d(const double* v) : x(v[0]), y(v[1]), z(v[2]) {}
  PrintVector3d(Eigen::Vector3d v) : x(v[0]), y(v[1]), z(v[2]) {}
};

std::ostream& operator<<(std::ostream& stream, PrintVector3d v) {
  stream << '(';
  stream << v.x << ", ";
  stream << v.y << ", ";
  stream << v.z << ')';
  return stream;
}

template <typename T>
static void printMember(
    std::ostream& stream,
    const char* key,
    const T& value,
    const char* valueColor = "\e[38;5;227m") {
  stream << "\n    \e[38;5;225m" << key;
  stream << "\e[0m \e[38;5;240m=\e[0m " << valueColor << value;
  stream << "\e[0m";
}

template <typename T>
static void printMemberWithUnits(
    std::ostream& stream, const char* key, const T& value, const char* units) {
  printMember(stream, key, value);
  stream << " \e[38;5;208m[" << units << "]\e[0m";
}

static void printMemberString(
    std::ostream& stream, const char* key, const std::string& value) {
  if (not value.empty())
    printMember(stream, key, "“" + value + "”", "\e[38;5;226m");
}

struct TaskView {
  struct Sentinel {};
  bool operator==(Sentinel) const {
    return task == binFile->tasks.data() + binFile->tasks.size();
  }
  bool operator!=(Sentinel) const { return not operator==(Sentinel()); }
  TaskView& operator*() noexcept { return *this; }
  TaskView& operator++() noexcept {
    ++task;
    return *this;
  }
  std::string printString() const {
    std::ostringstream stream;
    int taskIndex = int(task - binFile->tasks.data());
    int taskCount = binFile->tasks.size();
    stream << std::setprecision(4);
    stream << "\e[1mTask " << taskIndex + 1 << " of " << taskCount << ":\e[0m";
#define PRINT_MEMBER(member) printMember(stream, #member, task->member)
#define PRINT_MEMBER_WITH_UNITS(member, units) \
  printMemberWithUnits(stream, #member, task->member, units)
#define PRINT_MEMBER_STRING(member) \
  printMemberString(stream, #member, charsToString(task->member))
    PRINT_MEMBER_STRING(description);
    PRINT_MEMBER_STRING(startDateTime);
    PRINT_MEMBER_STRING(stopDateTime);
    PRINT_MEMBER_WITH_UNITS(focalLength, "mm");
    PRINT_MEMBER_WITH_UNITS(pulseRepetitionFreq, "Hz");
    PRINT_MEMBER_WITH_UNITS(pulseDuration, "sec");
    PRINT_MEMBER_WITH_UNITS(pulseEnergy, "J");
    PRINT_MEMBER_WITH_UNITS(laserSpectralCenter, "μm");
    PRINT_MEMBER_WITH_UNITS(laserSpectralWidth, "μm");
    PRINT_MEMBER(pulseCount);
#undef PRINT_MEMBER
#undef PRINT_MEMBER_WITH_UNITS
#undef PRINT_MEMBER_STRING
    stream << '\n';
    return stream.str();
  }
  std::shared_ptr<BinFile> binFile = nullptr;
  BinFile::Task* task = nullptr;
};

struct PulseView {
  struct Sentinel {};
  bool operator==(Sentinel) const {
    return pulseHeader.pulseIndex >= task->pulseCount;
  }
  bool operator!=(Sentinel) const { return not operator==(Sentinel()); }
  PulseView& operator*() noexcept { return *this; }
  PulseView& operator++() noexcept {
    ++pulseHeader.pulseIndex;
    if (pulseHeader.pulseIndex < task->pulseCount)
      pulseHeader = binFile->loadPulseHeader(*task, pulseHeader.pulseIndex);
    waveformArray.clear();
    return *this;
  }
  int64_t size() const {
    int64_t xSize = binFile->fileHeader.xDetectorCount;
    int64_t ySize = binFile->fileHeader.yDetectorCount;
    int64_t sSize = pulseHeader.samplesPerTimeBin;
    return xSize * ySize * sSize;
  }
  EigenVectorView waveformView(int x, int y, int sampleIndex = 0) {
    if (x < 0 or y < 0 or  //
        x >= int(binFile->fileHeader.xDetectorCount) or
        y >= int(binFile->fileHeader.yDetectorCount))
      throw std::invalid_argument("Index out of range");
    if (waveformArray.empty())  // Loaded yet?
      waveformArray = binFile->loadPulse(*task, pulseHeader);
    double* ptr = waveformArray.data();
    int64_t xSize = binFile->fileHeader.xDetectorCount;
    int64_t ySize = binFile->fileHeader.yDetectorCount;
    int64_t tSize = pulseHeader.timeGateBinCount + 1;
    int64_t sSize = pulseHeader.samplesPerTimeBin;
    ptr += x * ySize * tSize * sSize;
    ptr += y * tSize * sSize;
    ptr += sSize;  // Skip first entry
    ptr += sampleIndex;
    return EigenVectorView(ptr, tSize - 1, Eigen::InnerStride<>(sSize));
  }
  Eigen::Vector3d rayOrigin() {
    const auto& fileHeader = binFile->fileHeader;
    Eigen::Vector3d rayOrigin = {0, 0, 0};
    Eigen::Map<Eigen::Matrix4d> receiverToPlatform(
        &pulseHeader.receiverMountToPlatform[0]);
    rayOrigin = receiverToPlatform.block<3, 3>(0, 0) * rayOrigin;
    rayOrigin += Eigen::Map<Eigen::Vector3d>(&pulseHeader.platformLocation[0]);
    return rayOrigin;
  }
  Eigen::Vector3d rayDirection(int x, int y) {
    const auto& fileHeader = binFile->fileHeader;
    Eigen::Vector3d rayDirection = {
        fileHeader.xDetectorPitch * x + fileHeader.xArrayOffset,
        fileHeader.yDetectorPitch * y + fileHeader.yArrayOffset,
        -task->focalLength * 1e3  // mm to micron
    };
    rayDirection = rotateYZX(pulseHeader.receiverMountPointing) * rayDirection;
    Eigen::Map<Eigen::Matrix4d> receiverToPlatform(
        &pulseHeader.receiverMountToPlatform[0]);
    rayDirection = receiverToPlatform.block<3, 3>(0, 0) * rayDirection +
                   receiverToPlatform.col(3).head(3);
    rayDirection = rotateYZX(pulseHeader.platformRotation) * rayDirection;
    rayDirection.normalize();
    return rayDirection;
  }
  std::string printString() const {
    std::ostringstream stream;
    int taskIndex = task - binFile->tasks.data();
    int pulseIndex = pulseHeader.pulseIndex;
    int pulseCount = task->pulseCount;
    stream << std::setprecision(4);
    stream << "\e[1mPulse " << pulseIndex + 1 << " of " << pulseCount
           << ":\e[0m";
    stream << " \e[2m(in Task " << taskIndex + 1 << ")\e[0m";
#define PRINT_MEMBER(member) printMember(stream, #member, pulseHeader.member)
#define PRINT_MEMBER_WITH_UNITS(member, units) \
  printMemberWithUnits(stream, #member, pulseHeader.member, units)
    PRINT_MEMBER_WITH_UNITS(pulseTime, "sec");
    PRINT_MEMBER_WITH_UNITS(timeGateStart, "sec");
    PRINT_MEMBER_WITH_UNITS(timeGateStop, "sec");
    PRINT_MEMBER(timeGateBinCount);
    PRINT_MEMBER(samplesPerTimeBin);
    printMemberWithUnits(
        stream, "platformLocation", PrintVector3d{pulseHeader.platformLocation},
        "m");
    printMemberWithUnits(
        stream, "platformRotation", PrintVector3d{pulseHeader.platformRotation},
        "rad");
    printMember(
        stream, "Is pulse compressed?",
        pulseHeader.dataCompression == 0 ? "no" : "yes");
    printMember(
        stream, "Is pulse loaded?", waveformArray.empty() ? "no" : "yes");
#undef PRINT_MEMBER
#undef PRINT_MEMBER_WITH_UNITS
    stream << '\n';
    return stream.str();
  }

  std::shared_ptr<BinFile> binFile = nullptr;
  BinFile::Task* task = nullptr;
  BinFile::PulseHeader pulseHeader;
  std::vector<double> waveformArray;
};

struct WaveformView {
  PulseView* pulse = nullptr;
  int xIndex = 0;
  int yIndex = 0;
  int sampleIndex = 0;
  EigenVectorView waveform;
  Eigen::Vector3d rayOrigin;
  Eigen::Vector3d rayDirection;
  bool done = false;

  WaveformView(PulseView* pulse, int x, int y, int sampleIndex)
      : pulse(pulse),
        xIndex(x),
        yIndex(y),
        sampleIndex(sampleIndex),
        waveform(pulse->waveformView(x, y, sampleIndex)),
        rayOrigin(pulse->rayOrigin()),
        rayDirection(pulse->rayDirection(x, y)) {}
  int size() const { return waveform.size(); }
  double getitem(int index) const { return waveform[index]; }
  double distanceToTime(double dist) const { return dist / speedOfLight * 2; }
  double timeToDistance(double time) const { return time * speedOfLight / 2; }
  double minDistance() const {
    return timeToDistance(pulse->pulseHeader.timeGateStart);
  }
  double maxDistance() const {
    return timeToDistance(pulse->pulseHeader.timeGateStop);
  }
  double intensityAt(double dist) const {
    int n = waveform.size();
    double pulseTime = pulse->pulseHeader.pulseTime;
    double timeGate0 = pulse->pulseHeader.timeGateStart;  //- pulseTime;
    double timeGate1 = pulse->pulseHeader.timeGateStop;   // - pulseTime;
    double time = distanceToTime(dist);
    if (not(time > timeGate0)) return 0;
    if (not(time < timeGate1)) return 0;
    double t = (time - timeGate0) / (timeGate1 - timeGate0) * n;
    int i0 = std::max(0, std::min(int(t), n - 1));
    int i1 = std::max(0, std::min(i0 + 1, n - 1));
    t = t - i0;
    return (1 - t) * waveform[i0] + t * waveform[i1];
  }
  Eigen::Vector3d pointAt(double dist) const {
    return rayOrigin + rayDirection * dist;
  }
  double smooth(int index) const {
    double numer = 0;
    double denom = 0;
    for (int i = index - 2; i <= index + 2; i++)
      if (i >= 0 and i < int(waveform.size())) {
        numer += waveform[i];
        denom += 1;
      }
    return numer / denom;
  }
  std::vector<double> smoothDerivative() const {
    std::vector<double> deriv(waveform.size() - 1);
    for (int index = 0; index + 1 < int(waveform.size()); index++)
      deriv[index] = (smooth(index + 1) - smooth(index));
    return deriv;
  }
  std::vector<double> findPeaks(double threshold = 2) const {
    if (not(threshold > 0))
      threshold = 0.25 * (*std::max_element(waveform.begin(), waveform.end()));
    double pulseTime = pulse->pulseHeader.pulseTime;
    double timeGate0 = pulse->pulseHeader.timeGateStart;  // - pulseTime;
    double timeGate1 = pulse->pulseHeader.timeGateStop;   // - pulseTime;
    double timeBinDuration =
        (timeGate1 - timeGate0) / (pulse->pulseHeader.timeGateBinCount);
    std::vector<double> peaks;
    auto deriv = smoothDerivative();
    for (int i = 1; i < int(deriv.size()); i++)
      if (deriv[i - 1] > 0 and deriv[i] < 0 and waveform[i] > threshold) {
        peaks.push_back(timeToDistance(timeGate0 + timeBinDuration * i));
      }
    return peaks;
  }
  struct Sentinel {};
  bool operator==(Sentinel) const noexcept { return done; }
  bool operator!=(Sentinel) const noexcept { return not done; }
  WaveformView& operator*() { return *this; }
  WaveformView& operator++() {
    xIndex++;
    if (xIndex >= pulse->binFile->fileHeader.xDetectorCount) {
      xIndex = 0;
      yIndex++;
      if (yIndex >= pulse->binFile->fileHeader.yDetectorCount) {
        yIndex = 0;
        done = true;
        return *this;
      }
    }
    rayDirection = pulse->rayDirection(xIndex, yIndex);
    waveform = pulse->waveformView(xIndex, yIndex);
    return *this;
  }
  std::string printString() const {
    std::ostringstream stream;
    int taskIndex = pulse->task - pulse->binFile->tasks.data();
    int pulseIndex = pulse->pulseHeader.pulseIndex;
    int xCount = pulse->binFile->fileHeader.xDetectorCount;
    int yCount = pulse->binFile->fileHeader.yDetectorCount;
    int sampleCount = pulse->pulseHeader.samplesPerTimeBin;
    int waveformIndex =
        xIndex * yCount * sampleCount + yIndex * sampleCount + sampleIndex;
    int waveformCount = xCount * yCount * sampleCount;
    stream << std::setprecision(4);
    stream << "\e[1mWaveform " << waveformIndex + 1 << " of " << waveformCount
           << ":\e[0m"
           << " \e[2m(in Pulse " << pulse->pulseHeader.pulseIndex + 1
           << ", in Task " << taskIndex + 1 << ")\e[0m";
    printMember(stream, "xIndex", xIndex);
    printMember(stream, "yIndex", yIndex);
    printMember(stream, "sampleIndex", sampleIndex);
    printMemberWithUnits(stream, "rayOrigin", PrintVector3d(rayOrigin), "m");
    printMember(stream, "rayDirection", PrintVector3d(rayDirection));
    stream << '\n';
    return stream.str();
  }
};

PYBIND11_MODULE(d5lidar, module) {
  module.doc() = "DIRSIG5 Lidar Python utilities.";

  py::class_<BinFile, std::shared_ptr<BinFile>> binFile(module, "BinFile");
  py::class_<TaskView> taskView(binFile, "Task");
  py::class_<PulseView> pulseView(taskView, "Pulse");
  py::class_<WaveformView> waveformView(pulseView, "Waveform");

  binFile
      .def(py::init<const std::string&>(), "filename"_a)
#define DEF_MEMBER(name, doc) \
  def_property_readonly(      \
      #name, [](BinFile& self) { return self.fileHeader.name; }, doc)
      .DEF_MEMBER(
          sceneOriginLatitude,
          "The latitude of scene origin in degrees using +East [-180,180)")
      .DEF_MEMBER(
          sceneOriginLongitude,
          "The longitude of scene origin in degrees using +North [-90,90]")
      .DEF_MEMBER(
          sceneOriginHeight,
          "The height of scene origin above WGS84 reference ellipsoid [m]")
      .DEF_MEMBER(
          xDetectorCount, "The number of detector elements in the X dimension.")
      .DEF_MEMBER(
          yDetectorCount, "The number of detector elements in the Y dimension.")
      .DEF_MEMBER(
          xDetectorPitch,
          "The distance between detector centers [μm] in X dimension.")
      .DEF_MEMBER(
          yDetectorPitch,
          "The distance between detector centers [μm] in Y dimension.")
      .DEF_MEMBER(
          xArrayOffset,
          "The distance between array center and optical axis [μm].")
      .DEF_MEMBER(
          yArrayOffset,
          "The distance between array center and optical axis [μm].")
      .DEF_MEMBER(lensK1, "The first radial lens distortion coefficient k1.")
      .DEF_MEMBER(lensK2, "The second radial lens distortion coefficient k2.")
#undef DEF_MEMBER
      .def("__len__", [](BinFile& self) { return self.fileHeader.taskCount; })
      .def(
          "__getitem__",
          [](std::shared_ptr<BinFile> self, size_t taskIndex) {
            return TaskView{self, &self->tasks[taskIndex]};
          })
      .def("__iter__", [](std::shared_ptr<BinFile> self) {
        TaskView from = {self, self->tasks.data()};
        TaskView::Sentinel to;
        return py::make_iterator(from, to);
      });

  // BinFile.Task
  taskView
#define DEF_MEMBER(name, doc) \
  def_property_readonly(      \
      #name, [](TaskView& self) { return self.task->name; }, doc)
      .DEF_MEMBER(focalLength, "The instrument focal length [mm].")
      .DEF_MEMBER(
          pulseRepetitionFreq,
          "The nominal laser pulse repetition frequency [Hz].")
      .DEF_MEMBER(pulseDuration, "The nominal laser pulse duration [sec].")
      .DEF_MEMBER(pulseEnergy, "The nominal laser pulse energy [J].")
      .DEF_MEMBER(laserSpectralCenter, "The laser spectral center [μm].")
      .DEF_MEMBER(laserSpectralWidth, "The laser spectral width [μm].")
      .DEF_MEMBER(pulseCount, "The number of pulses in this task.")
#undef DEF_MEMBER
      .def("__str__", &TaskView::printString)
      .def("__len__", [](TaskView& self) { return self.task->pulseCount; })
      .def(
          "__getitem__",
          [](TaskView& self, size_t pulseIndex) {
            return PulseView{
                self.binFile,
                self.task,
                self.binFile->loadPulseHeader(*self.task, pulseIndex),
                {}};
          })
      .def("__iter__", [](TaskView& self) {
        PulseView from = {
            self.binFile, self.task,
            self.binFile->loadPulseHeader(*self.task, 0)};
        PulseView::Sentinel to;
        return py::make_iterator(from, to);
      });

  // BinFile.Task.Pulse
  pulseView
#define DEF_MEMBER(name, doc) \
  def_property_readonly(      \
      #name, [](PulseView& self) { return self.pulseHeader.name; }, doc)
      .DEF_MEMBER(pulseTime, "The pulse time relative to task start [sec].")
      .DEF_MEMBER(
          timeGateStart, "The gating start time relative to task start [sec].")
      .DEF_MEMBER(
          timeGateStop, "The gating stop time relative to task start [sec].")
      .DEF_MEMBER(
          timeGateBinCount,
          "The number of time samples (equally spaced) over time gating.")
      .DEF_MEMBER(
          samplesPerTimeBin,
          "The number of samples (equally spaced) within a time bin.")
#undef DEF_MEMBER
#define DEF_VECTOR3(name, doc)                                                \
  def_property_readonly(                                                      \
      #name,                                                                  \
      [](PulseView& self) { return Eigen::Vector3d(self.pulseHeader.name); }, \
      doc)
#define DEF_MATRIX4(name, doc)                                                \
  def_property_readonly(                                                      \
      #name,                                                                  \
      [](PulseView& self) { return Eigen::Matrix4d(self.pulseHeader.name); }, \
      doc)
      .DEF_VECTOR3(
          platformLocation,
          "The platform location (x,y,z) in scene local coordinates [m]")
      .DEF_VECTOR3(
          platformRotation,
          "The platform Euler angles (x,y,z) applied in \"yzx\" order [rad]")
      .DEF_MATRIX4(
          transmitterToMount,
          "The Tx to mount transform (4x4 affine matrix, row major)")
      .DEF_VECTOR3(
          transmitterMountPointing,
          "The Tx pointing Euler angles (x,y,z) applied in \"yzx\" order [rad]")
      .DEF_MATRIX4(
          transmitterMountToPlatform,
          "The Tx mount to platform transform (4x4 matrix, row major)")
      .DEF_MATRIX4(
          receiverToMount,
          "The Rx to mount transform (4x4 affine matrix, row major)")
      .DEF_VECTOR3(
          receiverMountPointing,
          "The Rx pointing Euler angles (x,y,z) applied in \"yzx\" order [rad]")
      .DEF_MATRIX4(
          receiverMountToPlatform,
          "The Rx mount to platform transform (4x4 affine matrix, row major)")
      .DEF_MATRIX4(
          systemPolTransmit,
          "The system polarization, transmit (optics, scan mirrors, etc...)")
      .DEF_MATRIX4(
          systemPolReceive,
          "The system polarization, receive (optics, scan mirrors, etc...)")
      .def_property_readonly(
          "waveformArray",
          [](py::object obj) {
            auto& self = obj.cast<PulseView&>();
            if (self.waveformArray.empty())  // Loaded yet?
              self.waveformArray =
                  self.binFile->loadPulse(*self.task, self.pulseHeader);
            return py::array_t<double>(
                {py::ssize_t(self.binFile->fileHeader.xDetectorCount),
                 py::ssize_t(self.binFile->fileHeader.yDetectorCount),
                 py::ssize_t(self.pulseHeader.timeGateBinCount + 1),
                 py::ssize_t(self.pulseHeader.samplesPerTimeBin)},
                self.waveformArray.data(), obj);
          })
      .def(
          "waveform",
          [](PulseView& self, int x, int y, int sampleIndex) {
            return WaveformView(&self, x, y, sampleIndex);
          },
          "xIndex"_a, "yIndex"_a, "sampleIndex"_a = 0, py::keep_alive<0, 1>())
      .def_property_readonly(
          "rayOrigin", &PulseView::rayOrigin, "The line-of-sight ray origin.")
      .def(
          "rayDirection", &PulseView::rayDirection, "xIndex"_a, "yIndex"_a,
          "The line-of-sight ray direction as a function of pixel X and Y "
          "indexes.")
      .def("__str__", &PulseView::printString)
      .def("__len__", &PulseView::size)
      .def(
          "__iter__",
          [](PulseView& self) {
            WaveformView from = WaveformView(&self, 0, 0, 0);
            WaveformView::Sentinel to;
            return py::make_iterator(from, to, py::return_value_policy::copy);
          },
          py::keep_alive<0, 1>())
      .def(
          "__getitem__",
          [](PulseView& self, int index) {
            WaveformView from = WaveformView(&self, 0, 0, 0);
            WaveformView::Sentinel to;
            while (index > 0) {
              --index;
              ++from;
              if (from == to) throw std::invalid_argument("Index out of range");
            }
            return from;
          },
          py::keep_alive<0, 1>());

  waveformView  //
      .def_readonly("pulse", &WaveformView::pulse, "The associated pulse.")
      .def_readonly("xIndex", &WaveformView::xIndex, "The pixel X index.")
      .def_readonly("yIndex", &WaveformView::yIndex, "The pixel Y index.")
      .def_readonly(
          "sampleIndex", &WaveformView::sampleIndex, "The sample index.")
      .def_readonly(
          "rayOrigin", &WaveformView::rayOrigin,
          "The line-of-sight ray origin.")
      .def_readonly(
          "rayDirection", &WaveformView::rayDirection,
          "The line-of-sight ray direction.")
      .def_property_readonly(
          "minDistance", &WaveformView::minDistance,
          "The minimum observable distance [m].")
      .def_property_readonly(
          "maxDistance", &WaveformView::maxDistance,
          "The maximum observable distance [m].")
      .def_property_readonly(
          "time",
          [](WaveformView& self) { return self.pulse->pulseHeader.pulseTime; },
          "The time of the pulse relative to the task start [sec].")
      .def("timeToDistance", &WaveformView::timeToDistance)
      .def("distanceToTime", &WaveformView::distanceToTime)
      .def(
          "intensityAtDistance", &WaveformView::intensityAt, "dist"_a,
          "Evaluate the intensity as a function of distance [m].")
      .def(
          "pointAtDistance", &WaveformView::pointAt, "dist"_a,
          "Evaluate the points along the line of sight as a function of "
          "distance [m].")
      .def(
          "findPeakDistances", &WaveformView::findPeaks, "threshold"_a = 4,
          "Find the distances of waveform peaks along the line of sight.")
      .def("__str__", &WaveformView::printString)
      .def("__len__", &WaveformView::size)
      .def("__getitem__", &WaveformView::getitem)
      .def(
          "__iter__",
          [](WaveformView& self) {
            return py::make_iterator(
                self.waveform.begin(), self.waveform.end(),
                py::return_value_policy::copy);
          })
      .def_property_readonly("array", [](py::object obj) {
        auto& self = obj.cast<WaveformView&>();
        return py::array_t<double>(
            py::ssize_t(self.waveform.size()), self.waveform.data(), obj);
      });
}
