#include "cuda_check.h"
#include "kernels.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kBlockSize = 256;

// Параметры замеров времени.
constexpr int kWarmup = 3;
constexpr int kRepeats = 20;
constexpr int kBenchBlockSizes[] = {64, 128, 256, 512, 1024};

struct Options {
    std::string input;
    std::string output;
    AccessMode mode = AccessMode::Coalesced;
};

void print_help(const char *prog) {
    std::cout << "Usage: " << prog
              << " --input <path> --output <path> [--mode coalesced|noncoalesced]\n"
                 "\n"
                 "Computes add (x+y), SAXPY (alpha*x+y) and ReLU (max(0,x)) on the GPU.\n"
                 "\n"
                 "  --input <path>   input file: 'N alpha', then N pairs 'x y'\n"
                 "  --output <path>  output file: N lines 'x+y alpha*x+y max(0,x)'\n"
                 "  --mode <mode>    memory access: coalesced (default) or noncoalesced\n"
                 "  --help           show this message\n"
                 "\n"
                 "N=0 produces an empty output file.\n"
                 "After writing the output, timings for the selected mode are printed to stdout:\n"
                 "H2D, each kernel, D2H and the full H2D -> kernels -> D2H cycle for block sizes\n"
                 "64..1024 (3 warm-up runs, 20 measured runs, median/min/max in ms).\n"
                 "Exit codes: 0 success, 2 invalid arguments or input, 1 runtime error.\n";
}

// Неверные аргументы или входные данные: сообщение в stderr и код возврата 2.
[[noreturn]] void input_error(const std::string &message) {
    std::cerr << "error: " << message << "\n";
    std::exit(2);
}

Options parse_args(int argc, char **argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (i + 1 >= argc) {
            input_error("missing value for " + arg);
        }
        const std::string value = argv[++i];
        if (arg == "--input") {
            opt.input = value;
        } else if (arg == "--output") {
            opt.output = value;
        } else if (arg == "--mode" && value == "coalesced") {
            opt.mode = AccessMode::Coalesced;
        } else if (arg == "--mode" && value == "noncoalesced") {
            opt.mode = AccessMode::NonCoalesced;
        } else if (arg == "--mode") {
            input_error("unknown mode: " + value);
        } else {
            input_error("unknown argument: " + arg);
        }
    }
    if (opt.input.empty() || opt.output.empty()) {
        input_error("--input and --output are required (see --help)");
    }
    return opt;
}

// Читает число из файла; мусор вроде "1.5abc" считается ошибкой входа.
float read_float(std::ifstream &in) {
    std::string token;
    if (!(in >> token)) {
        input_error("unexpected end of input");
    }
    char *end = nullptr;
    const float value = std::strtof(token.c_str(), &end);
    if (end == token.c_str() || *end != '\0') {
        input_error("invalid number: " + token);
    }
    return value;
}

void read_input(const std::string &path, float &alpha, std::vector<float> &x,
                std::vector<float> &y) {
    std::ifstream in(path);
    if (!in) {
        input_error("cannot open input file: " + path);
    }
    long long n = 0;
    if (!(in >> n) || n < 0) {
        input_error("expected non-negative N at the start of the input");
    }
    alpha = read_float(in);
    x.resize(n);
    y.resize(n);
    for (long long i = 0; i < n; ++i) {
        x[i] = read_float(in);
        y[i] = read_float(in);
    }
    std::string extra;
    if (in >> extra) {
        input_error("unexpected data after " + std::to_string(n) + " pairs");
    }
}

void run_gpu(float alpha, const std::vector<float> &x, const std::vector<float> &y, AccessMode mode,
             std::vector<float> &add, std::vector<float> &saxpy, std::vector<float> &relu) {
    const std::size_t n = x.size();
    const std::size_t bytes = n * sizeof(float);
    add.resize(n);
    saxpy.resize(n);
    relu.resize(n);
    if (n == 0) {
        return;
    }

    float *d_x = nullptr, *d_y = nullptr, *d_add = nullptr, *d_saxpy = nullptr, *d_relu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMalloc(&d_add, bytes));
    CUDA_CHECK(cudaMalloc(&d_saxpy, bytes));
    CUDA_CHECK(cudaMalloc(&d_relu, bytes));

    CUDA_CHECK(cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, y.data(), bytes, cudaMemcpyHostToDevice));

    launch_add(d_x, d_y, d_add, n, mode, kBlockSize);
    launch_saxpy(alpha, d_x, d_y, d_saxpy, n, mode, kBlockSize);
    launch_relu(d_x, d_relu, n, mode, kBlockSize);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(add.data(), d_add, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(saxpy.data(), d_saxpy, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(relu.data(), d_relu, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_add));
    CUDA_CHECK(cudaFree(d_saxpy));
    CUDA_CHECK(cudaFree(d_relu));
}

double ms_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
        .count();
}

// Печатает медиану, минимум и максимум одной величины: "bench block=256 metric=h2d ...".
void print_stats(int block, const char *metric, std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t m = values.size() / 2;
    const double median = values.size() % 2 == 1 ? values[m] : (values[m - 1] + values[m]) / 2.0;
    std::printf("bench block=%d metric=%s median_ms=%.4f min_ms=%.4f max_ms=%.4f\n", block, metric,
                median, values.front(), values.back());
}

// Замеры времени для выбранного режима доступа. cudaMemcpy из обычной (pageable) памяти
// синхронен относительно CPU, ядра асинхронны, поэтому времена H2D, ядер и D2H берём по
// событиям CUDA на GPU, а полный цикл — CPU-таймером после синхронизации с последним событием.
void benchmark(float alpha, const std::vector<float> &x, const std::vector<float> &y,
               AccessMode mode) {
    const std::size_t n = x.size();
    const std::size_t bytes = n * sizeof(float);

    cudaDeviceProp prop{};
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::printf("bench device=\"%s\" cc=%d.%d n=%zu mode=%s warmup=%d repeats=%d\n", prop.name,
                prop.major, prop.minor, n,
                mode == AccessMode::Coalesced ? "coalesced" : "noncoalesced", kWarmup, kRepeats);
    if (n == 0) {
        std::printf("bench skipped: N=0\n");
        return;
    }

    // Выделение памяти измеряется отдельно и в полный цикл не входит.
    float *d_x = nullptr, *d_y = nullptr, *d_add = nullptr, *d_saxpy = nullptr, *d_relu = nullptr;
    auto start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMalloc(&d_add, bytes));
    CUDA_CHECK(cudaMalloc(&d_saxpy, bytes));
    CUDA_CHECK(cudaMalloc(&d_relu, bytes));
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("bench alloc_ms=%.4f\n", ms_since(start));

    std::vector<float> add(n), saxpy(n), relu(n);

    // Точки: 0 — старт, 1 — после H2D, 2/3/4 — после add/saxpy/relu, 5 — после D2H.
    cudaEvent_t ev[6];
    for (cudaEvent_t &e : ev) {
        CUDA_CHECK(cudaEventCreate(&e));
    }

    for (int block : kBenchBlockSizes) {
        std::vector<double> h2d, k_add, k_saxpy, k_relu, kernels, d2h, full;
        for (int iter = 0; iter < kWarmup + kRepeats; ++iter) {
            start = std::chrono::steady_clock::now();
            CUDA_CHECK(cudaEventRecord(ev[0]));
            CUDA_CHECK(cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_y, y.data(), bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaEventRecord(ev[1]));
            launch_add(d_x, d_y, d_add, n, mode, block);
            CUDA_CHECK(cudaEventRecord(ev[2]));
            launch_saxpy(alpha, d_x, d_y, d_saxpy, n, mode, block);
            CUDA_CHECK(cudaEventRecord(ev[3]));
            launch_relu(d_x, d_relu, n, mode, block);
            CUDA_CHECK(cudaEventRecord(ev[4]));
            CUDA_CHECK(cudaMemcpy(add.data(), d_add, bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(saxpy.data(), d_saxpy, bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(relu.data(), d_relu, bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaEventRecord(ev[5]));
            CUDA_CHECK(cudaEventSynchronize(ev[5]));
            const double full_ms = ms_since(start);

            if (iter < kWarmup) {
                continue;
            }
            float ms[5];
            for (int i = 0; i < 5; ++i) {
                CUDA_CHECK(cudaEventElapsedTime(&ms[i], ev[i], ev[i + 1]));
            }
            h2d.push_back(ms[0]);
            k_add.push_back(ms[1]);
            k_saxpy.push_back(ms[2]);
            k_relu.push_back(ms[3]);
            kernels.push_back(ms[1] + ms[2] + ms[3]);
            d2h.push_back(ms[4]);
            full.push_back(full_ms);
        }
        print_stats(block, "h2d", h2d);
        print_stats(block, "add", k_add);
        print_stats(block, "saxpy", k_saxpy);
        print_stats(block, "relu", k_relu);
        print_stats(block, "kernels", kernels);
        print_stats(block, "d2h", d2h);
        print_stats(block, "full", full);
    }

    for (cudaEvent_t &e : ev) {
        CUDA_CHECK(cudaEventDestroy(e));
    }
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_add));
    CUDA_CHECK(cudaFree(d_saxpy));
    CUDA_CHECK(cudaFree(d_relu));
}

void write_output(const std::string &path, const std::vector<float> &add,
                  const std::vector<float> &saxpy, const std::vector<float> &relu) {
    FILE *f = std::fopen(path.c_str(), "w");
    if (f == nullptr) {
        std::cerr << "error: cannot open output file: " << path << "\n";
        std::exit(1);
    }
    for (std::size_t i = 0; i < add.size(); ++i) {
        std::fprintf(f, "%.9g %.9g %.9g\n", add[i], saxpy[i], relu[i]);
    }
    if (std::ferror(f) || std::fclose(f) != 0) {
        std::cerr << "error: failed to write output file: " << path << "\n";
        std::exit(1);
    }
}

} // namespace

int main(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--help") {
            print_help(argv[0]);
            return 0;
        }
    }
    const Options opt = parse_args(argc, argv);

    float alpha = 0.0f;
    std::vector<float> x, y;
    read_input(opt.input, alpha, x, y);

    std::vector<float> add, saxpy, relu;
    run_gpu(alpha, x, y, opt.mode, add, saxpy, relu);

    write_output(opt.output, add, saxpy, relu);

    benchmark(alpha, x, y, opt.mode);
    return 0;
}
