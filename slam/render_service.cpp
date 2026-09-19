#include "render_service.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

RenderService::RenderService(int port)
{
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0)
        return;
    int one = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (::bind(listen_fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0 || ::listen(listen_fd_, 16) < 0)
    {
        std::perror("render service: cannot listen (port in use? a stale trainer from an earlier run?)");
        ::close(listen_fd_);
        listen_fd_ = -1;
        std::exit(2); // never train unpaced by accident
    }
    ::fcntl(listen_fd_, F_SETFL, ::fcntl(listen_fd_, F_GETFL) | O_NONBLOCK); // accept() never waits
    std::printf("render service listening on 127.0.0.1:%d; waiting for a pacing client\n", port);
}

RenderService::~RenderService()
{
    if (listen_fd_ >= 0)
        ::close(listen_fd_);
}

int RenderService::poll(const Handler &handler)
{
    int served = 0;
    if (listen_fd_ < 0)
        return served;
    for (;;)
    {
        int fd = ::accept(listen_fd_, nullptr, nullptr);
        if (fd < 0)
            return served; // queue empty
        ::fcntl(fd, F_SETFL, ::fcntl(fd, F_GETFL) & ~O_NONBLOCK);
        char magic[4];
        if (readAll(fd, magic, 4))
        {
            if (std::memcmp(magic, "PACE", 4) == 0)
            {
                int32_t allowed;
                if (readAll(fd, &allowed, sizeof(allowed)))
                {
                    allowed_frame_ = allowed;
                    writeAll(fd, &current_frame_, sizeof(current_frame_));
                    served++;
                }
            }
            else if (std::memcmp(magic, "FIT2", 4) == 0)
            {
                int32_t head[2] = {fit_round_, static_cast<int32_t>(fit_maps_.size())};
                bool sent = writeAll(fd, head, sizeof(head));
                for (size_t i = 0; sent && i < fit_maps_.size(); i++)
                {
                    const FitMap &m = fit_maps_[i];
                    int32_t info[3] = {m.frame_id, m.width, m.height};
                    sent = writeAll(fd, info, sizeof(info)) && writeAll(fd, m.ssim.data(), m.ssim.size() * sizeof(float)) &&
                           writeAll(fd, m.depth.data(), m.depth.size() * sizeof(float));
                }
                if (sent)
                    served++;
            }
            else if (std::memcmp(magic, "NVS1", 4) == 0)
            {
                RenderRequest req;
                int32_t wh[2];
                float intr[4];
                if (readAll(fd, wh, sizeof(wh)) && readAll(fd, intr, sizeof(intr)) &&
                    readAll(fd, req.c2w, sizeof(req.c2w)) && wh[0] > 0 && wh[1] > 0)
                {
                    req.width = wh[0]; req.height = wh[1];
                    req.fx = intr[0]; req.fy = intr[1]; req.cx = intr[2]; req.cy = intr[3];
                    std::vector<uint8_t> rgb;
                    std::vector<float> depth;
                    if (handler(req, rgb, depth) &&
                        rgb.size() == static_cast<size_t>(req.width) * req.height * 3 &&
                        depth.size() == static_cast<size_t>(req.width) * req.height)
                    {
                        writeAll(fd, wh, sizeof(wh));
                        writeAll(fd, rgb.data(), rgb.size());
                        writeAll(fd, depth.data(), depth.size() * sizeof(float));
                        served++;
                    }
                }
            }
        }
        ::close(fd); // a closed connection without a reply tells the client the request was refused
    }
}

void RenderService::waitUntilAllowed(int frame, const Handler &handler)
{
    current_frame_ = frame;
    while (listen_fd_ >= 0 && allowed_frame_ < frame)
    {
        if (poll(handler) == 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

bool RenderService::readAll(int fd, void *buf, size_t n)
{
    auto *p = static_cast<uint8_t *>(buf);
    while (n)
    {
        ssize_t r = ::read(fd, p, n);
        if (r <= 0)
            return false;
        p += r;
        n -= static_cast<size_t>(r);
    }
    return true;
}

bool RenderService::writeAll(int fd, const void *buf, size_t n)
{
    auto *p = static_cast<const uint8_t *>(buf);
    while (n)
    {
        ssize_t r = ::write(fd, p, n);
        if (r <= 0)
            return false;
        p += r;
        n -= static_cast<size_t>(r);
    }
    return true;
}
