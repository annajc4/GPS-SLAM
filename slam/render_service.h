// Render service for GPS-SLAM: answers "render this camera" requests from another process while
// training runs, and paces training so it never runs ahead of that process (the dataset stands in
// for a live camera: "you may process frame N" carries the same information as "here is frame N").
// With the service enabled the trainer waits for the client from frame 0; a client must pace it.
// Drop this file and render_service.cpp into GPS-SLAM/slam/ (the CMake glob picks them up),
// apply render_service.patch, and set PIPE.render_service_port in the config.
//
// Everything is served from the training thread, between optimisation rounds or while waiting for
// permission to advance, so there is no concurrent access to the model. Renders follow
// remote_viewer.cpp / renderEvalImgs exactly: SDF raycast, then the Gaussian forward pass.
//
// Wire protocol (little-endian, one message per TCP connection):
//   "NVS1" render : int32 width, int32 height, float32 fx, fy, cx, cy,
//                   float32[16] camera-to-world (row-major, OpenCV axes, SLAM/first-frame coordinates)
//          reply  : int32 width, int32 height, uint8[h*w*3] RGB, float32[h*w] camera-z depth (metres)
//   "PACE" pace   : int32 allowed_frame  (training may process frames <= allowed_frame; INT32_MAX = no limit)
//          reply  : int32 current_frame  (the frame training is at or waiting to start)
#pragma once
#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

struct RenderRequest
{
    int width = 0, height = 0;
    float fx = 0, fy = 0, cx = 0, cy = 0;
    float c2w[16] = {};
};

class RenderService
{
public:
    // Renders into rgb (h*w*3, row-major RGB) and depth (h*w, metres). Return false to refuse.
    using Handler = std::function<bool(const RenderRequest &, std::vector<uint8_t> &rgb, std::vector<float> &depth)>;

    explicit RenderService(int port);
    ~RenderService();
    RenderService(const RenderService &) = delete;
    RenderService &operator=(const RenderService &) = delete;

    // Answer every queued message, then return immediately.
    int poll(const Handler &handler);

    // Block until a PACE message allows `frame`, serving messages meanwhile. A trainer with the
    // service enabled starts blocked at frame 0: it never processes a frame the client has not
    // released, so no render can contain frames the client has not seen.
    void waitUntilAllowed(int frame, const Handler &handler);

    bool ok() const { return listen_fd_ >= 0; }

private:
    int listen_fd_ = -1;
    int32_t allowed_frame_ = -1;
    int32_t current_frame_ = -1;
    static bool readAll(int fd, void *buf, size_t n);
    static bool writeAll(int fd, const void *buf, size_t n);
};
