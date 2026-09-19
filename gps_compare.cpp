// Offline comparison driver for MisEty/GPS-SLAM release.
// Uses the native training pipeline and hybrid rendering functions.
#include "dataset_reader.h"
#include "slam_pipeline.h"
#include <cuda_runtime_api.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

using Json = nlohmann::json;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

static double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
static void syncGPU() {
    auto error = cudaDeviceSynchronize();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
static void writeJson(const fs::path &path, const Json &value) {
    std::ofstream file(path);
    if (!file) throw std::runtime_error("Cannot write " + path.string());
    file << value.dump(2) << '\n';
    if (!file) throw std::runtime_error("Write failed: " + path.string());
}
static torch::Tensor poseTensor(const Json &matrix) {
    std::vector<float> values;
    for (const auto &row : matrix)
        for (const auto &value : row) values.push_back(value.get<float>());
    if (matrix.size() != 4 || values.size() != 16)
        throw std::runtime_error("Expected a 4x4 camera pose");
    return torch::from_blob(values.data(), {4, 4}, torch::kFloat32).clone();
}

// NumPy v1.0: float32, row-major, HxW camera-z depth in metres.
static void writeDepth(const fs::path &path, const torch::Tensor &input) {
    auto depth = input.detach().to(torch::kCPU).to(torch::kFloat32).squeeze(-1).contiguous();
    if (depth.dim() != 2) throw std::runtime_error("Depth must be HxWx1 or HxW");
    const uint16_t endian = 1;
    if (*reinterpret_cast<const uint8_t *>(&endian) != 1)
        throw std::runtime_error("NPY writer requires a little-endian host");
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
        std::to_string(depth.size(0)) + ", " + std::to_string(depth.size(1)) + "), }";
    header.append((64 - ((10 + header.size() + 1) % 64)) % 64, ' ');
    header += '\n';
    const uint16_t length = static_cast<uint16_t>(header.size());
    const char prefix[] = {char(0x93), 'N', 'U', 'M', 'P', 'Y', 1, 0,
                           char(length & 255), char(length >> 8)};
    std::ofstream file(path, std::ios::binary);
    file.write(prefix, sizeof(prefix));
    file.write(header.data(), header.size());
    file.write(reinterpret_cast<const char *>(depth.data_ptr<float>()), depth.numel() * sizeof(float));
    if (!file) throw std::runtime_error("Depth write failed: " + path.string());
}

static void train(const YAML::Node &config, const std::string &configFile) {
    const auto started = Clock::now();
    const std::string workspace = config["workspace_dir"].as<std::string>();
    if (fs::exists(workspace))
        throw std::runtime_error("Refusing to clear an existing workspace: " + workspace);
    if (config["work_mode"].as<std::string>() != "train" ||
        !config["PIPE"]["TSDF"]["use_gt_pose"].as<bool>() ||
        config["READER"]["start_frame"].as<int>() != 0 ||
        config["READER"]["frame_step"].as<int>() != 1 ||
        config["READER"]["test_split_interval"].as<int>() != -1)
        throw std::runtime_error("Expected full-sequence training with supplied poses");
    Json timing;
    auto phase = Clock::now();
    DatasetReader reader(config["READER"]);
    reader.read();
    reader.updateSceneGeo();
    const size_t expected = config["READER"]["end_frame"].as<int>() + 1;
    if (reader.train_vec.size() != expected)
        throw std::runtime_error("Not all input cameras were loaded");
    timing["input_loading_seconds"] = seconds(phase);

    phase = Clock::now();
    auto *engine = createTsdfEngine(reader, config["PIPE"]["TSDF"]);
    SLAMPipeline pipe;
    pipe.setTsdfEngine(engine);
    pipe.work_mode = "train";
    pipe.device_id = config["dev_id"].as<int>();
    SLAMGaussianModel model;
    model.loadConfig(config["MODEL"]);
    pipe.scene_scale = reader.scene_scale;
    // Stock helper clears the workspace. The fresh-path guard above is essential.
    createWorkSpace(configFile);
    pipe.loadConfig(config["PIPE"], workspace, true);
    syncGPU();
    timing["engine_and_model_setup_seconds"] = seconds(phase);

    phase = Clock::now();
    pipe.SLAMTrainCams(model, reader.train_vec);
    syncGPU();
    const double reconstruction = seconds(phase);
    if (pipe.curr_frame_id + 1 != static_cast<int>(expected))
        throw std::runtime_error("Reconstruction did not finish every frame");
    if (model.getGaussianNum() <= 0)
        throw std::runtime_error("The native pipeline produced no Gaussians");
    timing["frames"] = expected;
    timing["reconstruction_seconds"] = reconstruction;
    timing["reconstruction_fps"] = expected / reconstruction;
    timing["gaussians"] = model.getGaussianNum();
    timing["scope"] = "Synchronized full native SLAMTrainCams call, including its logging; excludes input preload, setup, saving, and offline rendering";

    phase = Clock::now();
    pipe.save(model, reader.getAllCams());
    pipe.saveEngine();
    reader.savePose(pipe.eval_path + "/pose");
    syncGPU();
    timing["saving_seconds"] = seconds(phase);
    timing["driver_total_seconds"] = seconds(started);
    writeJson(fs::path(workspace) / "comparison_timing.json", timing);
    std::cout << "\nCOMPARISON TIMING\n" << timing.dump(2) << std::endl;
}

static void render(const YAML::Node &config, const fs::path &manifestFile, const fs::path &output) {
    if (fs::exists(output)) throw std::runtime_error("Output already exists: " + output.string());
    std::ifstream input(manifestFile);
    Json manifest;
    input >> manifest;
    const auto &calib = manifest.at("camera");
    const int w = calib.at("width"), h = calib.at("height");
    auto imageShape = config["READER"]["image_shape"].as<std::vector<int>>();
    auto intrinsics = config["READER"]["intrinsics"].as<std::vector<float>>();
    if (w != 640 || h != 480 || imageShape != std::vector<int>{w, h})
        throw std::runtime_error("Expected matching 640x480 calibration");
    const char *keys[] = {"fx", "fy", "cx", "cy"};
    for (int i = 0; i < 4; ++i)
        if (std::abs(calib.at(keys[i]).get<float>() - intrinsics.at(i)) > 1e-4f)
            throw std::runtime_error("Manifest/config intrinsics mismatch");
    if (manifest.at("views").empty()) throw std::runtime_error("Empty camera manifest");
    if (config["MODEL"]["use_exposure"].as<bool>())
        throw std::runtime_error("Virtual-camera exposure policy must be defined before enabling exposure");

    auto phase = Clock::now();
    // One dummy RGB-D camera allocates the native engine at the correct resolution.
    // No input frames are processed; the saved SDF is loaded below.
    DatasetReader reader(config["READER"]);
    Camera dummy(w, h, intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3],
                 true, torch::eye(4, torch::kFloat32));
    dummy.image = torch::zeros({h, w, 3}, torch::kFloat32);
    dummy.depth = torch::zeros({h, w, 1}, torch::kFloat32);
    reader.train_vec.push_back(dummy);
    auto *engine = createTsdfEngine(reader, config["PIPE"]["TSDF"]);
    SLAMPipeline pipe;
    pipe.setTsdfEngine(engine);
    pipe.work_mode = "eval";
    pipe.device_id = config["dev_id"].as<int>();
    pipe.scene_scale = reader.scene_scale;
    const std::string workspace = config["workspace_dir"].as<std::string>();
    pipe.loadConfig(config["PIPE"], workspace, false);
    SLAMGaussianModel model;
    model.loadConfig(config["MODEL"]);
    model.loadParamsTensor(workspace + "/gs_model/model.pt");
    model.setParamsDevice(torch::kCUDA);
    pipe.loadEngine();
    if (model.getGaussianNum() <= 0) throw std::runtime_error("Empty saved Gaussian model");
    syncGPU();
    Json timing = {{"loading_seconds", seconds(phase)}, {"views", manifest.at("views").size()}};
    torch::NoGradGuard noGrad;
    const auto makeCamera = [&](const Json &view) {
        Camera cam(w, h, intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3],
                   false, poseTensor(view.at("c2w_gps")));
        cam.id = -1;  // Always use the supplied virtual pose, never an input-camera index.
        return cam;
    };
    const auto forward = [&](const Camera &cam) {
        auto sdf = pipe.runRaycastByCam(cam, false);
        auto result = model.forward(cam, sdf.at("depth_map"), sdf.at("color_map"));
        return TensorDict{{"rgb", torch::clamp(result.at("rgb"), 0, 1)},
                          {"depth", sdf.at("depth_map")}};
    };
    phase = Clock::now();
    for (int i = 0; i < 2; ++i) forward(makeCamera(manifest.at("views").at(0)));
    syncGPU();
    timing["warmup_seconds"] = seconds(phase);
    fs::create_directories(output);
    double renderSeconds = 0, writeSeconds = 0;
    for (const auto &view : manifest.at("views")) {
        const std::string stem = view.at("stem");
        if (stem.empty() || stem.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") != std::string::npos)
            throw std::runtime_error("Unsafe output filename");
        syncGPU();
        phase = Clock::now();
        auto result = forward(makeCamera(view));
        auto rgb = tensorToImage(result.at("rgb"));
        auto depth = result.at("depth").detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
        depth = torch::where(torch::isfinite(depth) & (depth > 0), depth, torch::zeros_like(depth));
        syncGPU();
        renderSeconds += seconds(phase);
        phase = Clock::now();
        if (!cv::imwrite((output / (stem + "_rgb.png")).string(), rgb))
            throw std::runtime_error("RGB write failed");
        writeDepth(output / (stem + "_depth.npy"), depth);
        writeSeconds += seconds(phase);
        std::cout << "Rendered " << stem << std::endl;
    }
    timing["render_and_download_seconds"] = renderSeconds;
    timing["image_and_depth_write_seconds"] = writeSeconds;
    timing["depth_kind"] = "Metric SDF camera-z depth, metres; zero denotes invalid depth";
    writeJson(output / "timing.json", timing);
    fs::copy_file(manifestFile, output / "manifest.json");
    std::cout << timing.dump(2) << std::endl;
}

int main(int argc, char **argv) {
    try {
        if (argc < 3) throw std::runtime_error("Usage: gps_compare train CONFIG | render CONFIG MANIFEST OUTPUT");
        auto config = YAML::LoadFile(argv[2]);
        const std::string device = config["dev_id"].as<std::string>();
        setenv("CUDA_VISIBLE_DEVICES", device.c_str(), 1);
        if (config["MODEL"]["render_method"].as<std::string>() != "ges")
            throw std::runtime_error("Expected native hybrid render_method: ges");
        const std::string command = argv[1];
        if (command == "train" && argc == 3) train(config, argv[2]);
        else if (command == "render" && argc == 5) render(config, argv[3], argv[4]);
        else throw std::runtime_error("Invalid command or argument count");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Comparison failed: " << error.what() << std::endl;
        return 1;
    }
}
