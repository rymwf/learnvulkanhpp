module;

#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>

export module common.orbit_camera;

import std;

namespace {

constexpr float pi = glm::pi<float>();
constexpr glm::vec3 worldUp{0.0f, 1.0f, 0.0f};

bool isFinite(glm::vec3 value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

}  // namespace

export struct OrbitCameraConfig {
    glm::vec3 target{0.0f, 0.0f, 0.0f};
    float yawRadians = 3.0f * pi / 4.0f;
    float pitchRadians = pi / 9.0f;
    float distance = 5.3f;
    float minDistance = 2.0f;
    float maxDistance = 12.0f;
    float minPitchRadians = -4.0f * pi / 9.0f;
    float maxPitchRadians = 4.0f * pi / 9.0f;
    float zoomSensitivity = 0.12f;
};

export class OrbitCamera {
public:
    explicit OrbitCamera(OrbitCameraConfig config = {})
        : initialConfig_(validateConfig(config)),
          target_(initialConfig_.target),
          yawRadians_(initialConfig_.yawRadians),
          pitchRadians_(initialConfig_.pitchRadians),
          distance_(initialConfig_.distance) {}

    void orbit(float yawDeltaRadians, float pitchDeltaRadians) {
        if (!std::isfinite(yawDeltaRadians) || !std::isfinite(pitchDeltaRadians)) {
            throw std::invalid_argument("Orbit deltas must be finite");
        }
        yawRadians_ += yawDeltaRadians;
        pitchRadians_ = std::clamp(pitchRadians_ + pitchDeltaRadians, initialConfig_.minPitchRadians,
                                   initialConfig_.maxPitchRadians);
    }

    void pan(float horizontalPixels, float verticalPixels, float verticalFovRadians, float viewportHeightPixels) {
        if (!std::isfinite(horizontalPixels) || !std::isfinite(verticalPixels) ||
            !std::isfinite(verticalFovRadians) || !std::isfinite(viewportHeightPixels) ||
            verticalFovRadians <= 0.0f || verticalFovRadians >= pi || viewportHeightPixels <= 0.0f) {
            throw std::invalid_argument("Invalid orbit camera pan parameters");
        }

        const auto basis = cameraBasis();
        const auto worldUnitsPerPixel =
            2.0f * distance_ * std::tan(verticalFovRadians * 0.5f) / viewportHeightPixels;
        target_ += basis.right * (-horizontalPixels * worldUnitsPerPixel) +
                   basis.up * (verticalPixels * worldUnitsPerPixel);
    }

    void zoom(float scrollYOffset) {
        if (!std::isfinite(scrollYOffset)) {
            throw std::invalid_argument("Orbit camera scroll delta must be finite");
        }
        distance_ = std::clamp(distance_ * std::exp(-initialConfig_.zoomSensitivity * scrollYOffset),
                               initialConfig_.minDistance, initialConfig_.maxDistance);
    }

    void reset() noexcept {
        target_ = initialConfig_.target;
        yawRadians_ = initialConfig_.yawRadians;
        pitchRadians_ = initialConfig_.pitchRadians;
        distance_ = initialConfig_.distance;
    }

    [[nodiscard]] glm::vec3 position() const {
        const auto cosPitch = std::cos(pitchRadians_);
        return target_ + glm::vec3{
                             distance_ * cosPitch * std::cos(yawRadians_),
                             distance_ * std::sin(pitchRadians_),
                             distance_ * cosPitch * std::sin(yawRadians_),
                         };
    }

    [[nodiscard]] glm::vec3 target() const noexcept {
        return target_;
    }

    [[nodiscard]] float distance() const noexcept {
        return distance_;
    }

    [[nodiscard]] glm::mat4 viewMatrix() const {
        return glm::lookAtRH(position(), target_, worldUp);
    }

private:
    struct CameraBasis {
        glm::vec3 right;
        glm::vec3 up;
    };

    static OrbitCameraConfig validateConfig(OrbitCameraConfig config) {
        if (!isFinite(config.target) || !std::isfinite(config.yawRadians) || !std::isfinite(config.pitchRadians) ||
            !std::isfinite(config.distance) || !std::isfinite(config.minDistance) ||
            !std::isfinite(config.maxDistance) || !std::isfinite(config.minPitchRadians) ||
            !std::isfinite(config.maxPitchRadians) || !std::isfinite(config.zoomSensitivity)) {
            throw std::invalid_argument("Orbit camera configuration must be finite");
        }
        if (config.minDistance <= 0.0f || config.maxDistance < config.minDistance ||
            config.distance < config.minDistance || config.distance > config.maxDistance) {
            throw std::invalid_argument("Orbit camera distance limits are invalid");
        }
        if (config.minPitchRadians >= config.maxPitchRadians || config.pitchRadians < config.minPitchRadians ||
            config.pitchRadians > config.maxPitchRadians) {
            throw std::invalid_argument("Orbit camera pitch limits are invalid");
        }
        if (config.zoomSensitivity <= 0.0f) {
            throw std::invalid_argument("Orbit camera zoom sensitivity must be positive");
        }
        return config;
    }

    [[nodiscard]] CameraBasis cameraBasis() const {
        const auto forward = glm::normalize(target_ - position());
        const auto right = glm::normalize(glm::cross(forward, worldUp));
        return {
            .right = right,
            .up = glm::cross(right, forward),
        };
    }

    OrbitCameraConfig initialConfig_;
    glm::vec3 target_;
    float yawRadians_ = 0.0f;
    float pitchRadians_ = 0.0f;
    float distance_ = 0.0f;
};
