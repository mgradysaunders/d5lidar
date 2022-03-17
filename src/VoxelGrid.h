#include <Eigen/Geometry>

struct VoxelGrid {
  using Vec3f = Eigen::Vector3f;
  using Vec3i = Eigen::Vector3i;
  struct Bound {
    Vec3f points[2] = {};
    [[nodiscard]] const Vec3f& min() const noexcept { return points[0]; }
    [[nodiscard]] const Vec3f& max() const noexcept { return points[1]; }
    [[nodiscard]] Vec3f extent() const noexcept { return max() - min(); }
    [[nodiscard]] Vec3f center() const noexcept {
      return (max() + min()) * 0.5f;
    }
    [[nodiscard]] bool rayCast(
        const Vec3f& org,
        const Vec3f& dir,
        float& tmin,
        float& tmax) const noexcept {
      bool dirSignbit[3] = {
          std::signbit(dir[0]), std::signbit(dir[1]), std::signbit(dir[2])};
      tmin = (points[dirSignbit[0]][0] - org[0]) / dir[0];
      tmax = (points[!dirSignbit[0]][0] - org[0]) / dir[0];
      for (int k = 1; k < 3; k++) {
        float tmink = (points[dirSignbit[k]][k] - org[k]) / dir[k],
              tmaxk = (points[!dirSignbit[k]][k] - org[k]) / dir[k];
        if (not(tmin < tmaxk and tmax > tmink)) return false;
        tmin = std::max(tmin, tmink);
        tmax = std::min(tmax, tmaxk);
      }
      return true;
    }
    [[nodiscard]] bool contains(const Vec3f& coord) const noexcept {
      for (int i = 0; i < 3; i++)
        if (not(points[0][i] <= coord[i] and coord[i] <= points[1][i]))
          return false;
      return true;
    }
  };

  [[nodiscard]] Vec3i coordToIndex(Vec3f coord) const noexcept {
    for (int i = 0; i < 3; i++) {
      coord[i] -= bound.points[0][i];
      coord[i] /= bound.points[1][i] - bound.points[0][i];
      coord[i] *= count[i];
    }
    return {
        int(std::floor(coord[0])), int(std::floor(coord[1])),
        int(std::floor(coord[2]))};
  }

  [[nodiscard]] Vec3f getVoxelExtent() const noexcept {
    Vec3f extent = bound.extent();
    for (int i = 0; i < 3; i++) extent[i] /= count[i];
    return extent;
  }

  [[nodiscard]] Vec3f getVoxelCenter(Vec3i index) const noexcept {
    return getVoxelBound(index).center();
  }

  [[nodiscard]] Bound getVoxelBound(Vec3i index) const noexcept {
    Bound voxelBound;
    for (int i = 0; i < 3; i++) {
      voxelBound.points[0][i] = float(index[i]) / count[i];
      voxelBound.points[1][i] = float(index[i] + 1) / count[i];
    }
    return voxelBound;
  }

  template <typename Func>
  void traverse(Vec3f org, Vec3f dir, Func&& func) {
    Vec3i index;
    if (bound.contains(org))
      index = coordToIndex(org);
    else {
      float tmin = 0;
      float tmax = 0;
      if (!bound.rayCast(org, dir, tmin, tmax)) return;
      index = coordToIndex(org + dir * (tmin + 1e-4f));
    }
    while (true) {
      Bound voxelBound = getVoxelBound(index);
      float tmin = 0;
      float tmax = 0;
      if (voxelBound.rayCast(org, dir, tmin, tmax))
        std::invoke(std::forward<Func>(func), tmin, tmax, index);
      auto advanceToNext = [&] {
        for (int dim = 0; dim < 3; dim++) {
          Vec3i nextIndex = index;
          nextIndex[dim] += dir[dim] > 0 ? 1 : -1;
          if (nextIndex[dim] <= -1 or nextIndex[dim] >= count[dim]) continue;
          if (getVoxelBound(nextIndex).rayCast(org, dir, tmin, tmax)) {
            index = nextIndex;
            return true;
          }
        }
        return false;
      };
      if (not advanceToNext()) return;
    }
  }

  Vec3i count;
  Bound bound;
};
