/*
FMT_plug tool - PLENA
ver. 0.9

PROVENANCE / ATTRIBUTION

Author: Seok Minsoo / seokminsu10@yu.ac.kr

Author responsibility
- The author performed the C++ porting/implementation, experiments, analysis, and verification.
- The author takes full responsibility for the final code, results, interpretations, and manuscript.

1) MATLAB -> C++ port (used with permission)
- Core workflow (e.g., changedirect2 / loopcheck2 / Gibbs update loop, etc.) was ported to C++ based on a MATLAB reference implementation provided by Seo & Schmidt.
- C++ porting and modifications: Seok Minsoo (seokminsu10@yu.ac.kr).
- This repository includes a derivative implementation based on the provided reference code, used within the scope of the granted permission.

2) PRNG (third-party component)
- Uses the xoshiro256** reference implementation by David Blackman & Sebastiano Vigna.
- The upstream reference includes a public-domain-style dedication / broad permission notice and an “AS IS” (no-warranty) disclaimer.
- Reference: https://prng.di.unimi.it/xoshiro256starstar.c

3) Seed mixing (hash finalizer pattern / constants)
- Uses an fmix64-style finalizer pattern/constants commonly used with MurmurHash3.
- MurmurHash3 is widely distributed with a public domain disclaimer in the upstream source header.
- Reference example: https://github.com/rurban/smhasher/blob/master/MurmurHash3.cpp

4) AI assistance (transparent disclosure)
- The NSE (Nash–Sutcliffe Efficiency) function was written by a conversational AI tool.
- Multithreading: the author proposed the idea/direction. (Unless explicitly stated otherwise, implementation/coding is considered integrated and reviewed under the author’s responsibility.)
- In addition, limited assistance from a conversational AI tool was used for debugging suggestions, language polishing of text/comments, and code cleanup (readability-oriented refactoring).

5) Additional modifications by the author (examples)
- Batch execution (task scheduling, progress reporting), file I/O and file-selection UI, robust failure handling (e.g., CHANGE_MAX_TRIES), result writing (txt/csv), and reproducibility-related logging.
*/

///////////////////////////////헤더파일//////////////////////////////////////
#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <limits>
#include <chrono>
#include <queue>
#include <thread>
#include <atomic>
#include <mutex>
#include <exception>
#include <cstddef>   // std::size_t
#include <iomanip>   // setprecision, fixed, scientific
#include <map>       // beta(k)별 요약용

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

using namespace std;

namespace fs = filesystem;

//  옵션 
#define FIX_CELL_CONSTRAINT 0
static constexpr int FIX_I_1BASED = 11;
static constexpr int FIX_J_1BASED = 14;


// changedirect2 무한 루프 방지(실패 시 task 실패 처리)
static constexpr int CHANGE_MAX_TRIES = 200000;


//  Matrix (연속 메모리) 
struct 매트릭스
{
    int n = 0, m = 0;
    vector<int> data;
    매트릭스() = default;
    매트릭스(int rows, int cols, int value = 0) : n(rows), m(cols), data(rows* cols, value) {}
    inline int& operator()(int i, int j) { return data[i * m + j]; }
    inline const int& operator()(int i, int j) const { return data[i * m + j]; }
    void assign(int value) { fill(data.begin(), data.end(), value); }
};

//  시간 측정 
using Clock = chrono::steady_clock;
static inline Clock::time_point tic() { return Clock::now(); }
static inline double toc_sec(const Clock::time_point& t0) {
    return chrono::duration<double>(Clock::now() - t0).count();
}

//  문자열 ends_with 
static bool ends_with(const string& s, const string& suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

//  txt 파일 목록 
static vector<fs::path> listTxtFilesInCwd(bool excludeResultFiles = true) {
    vector<fs::path> files;
    fs::path cwd = fs::current_path();
    error_code ec;
    for (const auto& entry : fs::directory_iterator(cwd, ec)) {
        if (ec) break;
        if (!entry.is_regular_file()) continue;
        fs::path p = entry.path();
        if (p.extension() == ".txt") {
            string name = p.filename().string();
            if (excludeResultFiles && ends_with(name, "_결과.txt")) continue;
            files.push_back(p);
        }
    }
    sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
        error_code ec1, ec2;
        auto ta = fs::last_write_time(a, ec1);
        auto tb = fs::last_write_time(b, ec2);
        if (ec1 || ec2) return a.filename().string() < b.filename().string();
        return ta > tb;
        });
    return files;
}

//  파일 선택 UI 
static string chooseTxtFileInteractive() {
    auto files = listTxtFilesInCwd(true);
    
    cout << "CWD = " << fs::current_path().string() << "\n";
    
    if (files.empty())
    {
        cout << "현재 폴더에 선택 가능한 텍스트 포맷 파일이 없습니다.\n";
        return "";
    }

    cout << "\n//------ 입력 파일 목록 //------\n";
    for (size_t i = 0; i < files.size(); ++i)
    {
        cout << "[" << (i + 1) << "] " << files[i].filename().string() << "\n";
    }
    cout << "[0] 종료\n";
    
    while (true)
    {
        cout << "\n번호 선택: ";
        int sel;
        if (!(cin >> sel))
        {
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            continue;
        }
        if (sel == 0) return "";
        if (sel >= 1 && static_cast<size_t>(sel) <= files.size())
        {
            return files[sel - 1].string();
        }
    }
}

//  xoshiro256** PRNG <=명시 필요 
struct Xoshiro256ss
{
    uint64_t s[4];
    static uint64_t rotl(const uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    static uint64_t splitmix64(uint64_t& x) {
        uint64_t z = (x += 0x9E3779B97f4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    explicit Xoshiro256ss(uint64_t seed) {
        uint64_t x = seed;
        for (int i = 0; i < 4; ++i) s[i] = splitmix64(x);
    }
    uint64_t next() {
        const uint64_t result = rotl(s[1] * 5ULL, 7) * 9ULL;
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return result;
    }
    double uniform01(){ return (next() >> 11) * (1.0 / static_cast<double>(1ULL << 53)); }
    int uniformInt(int a, int b) {
        uint64_t r = next();
        return a + static_cast<int>(r % static_cast<uint64_t>(b - a + 1));
    }
};

// seed mixing (빠르고 괜찮은 분산)
static inline uint64_t mix_seed(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}



//  Flow Accumulation 
static void computeFlowAccum(const 매트릭스& D, const 매트릭스& I, 매트릭스& FA) {
    int n = D.n, m = D.m;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            int val = 0;
            if (j > 0 && D(i, j - 1) == 1) val += I(i, j - 1);
            if (i > 0 && D(i - 1, j) == 2) val += I(i - 1, j);
            if (j < m - 1 && D(i, j + 1) == 3) val += I(i, j + 1);
            if (i < n - 1 && D(i + 1, j) == 4) val += I(i + 1, j);
            FA(i, j) = val;
        }
    }
}

//  Travel Time (BFS) 
static void 도달시간계산(const 매트릭스& D, 매트릭스& T, int n0, int m0) {
    int n = D.n, m = D.m;
    T.assign(0);

    queue<int> q;
    T(n0, m0) = 1;
    q.push(n0 * m + m0);

    while (!q.empty()) {
        int pos = q.front(); q.pop();
        int i = pos / m;
        int j = pos % m;
        int curT = T(i, j);

        if (j > 0 && D(i, j - 1) == 1 && T(i, j - 1) == 0) {
            T(i, j - 1) = curT + 1;
            q.push(i * m + (j - 1));
        }
        if (i > 0 && D(i - 1, j) == 2 && T(i - 1, j) == 0) {
            T(i - 1, j) = curT + 1;
            q.push((i - 1) * m + j);
        }
        if (j < m - 1 && D(i, j + 1) == 3 && T(i, j + 1) == 0) {
            T(i, j + 1) = curT + 1;
            q.push(i * m + (j + 1));
        }
        if (i < n - 1 && D(i + 1, j) == 4 && T(i + 1, j) == 0) {
            T(i + 1, j) = curT + 1;
            q.push((i + 1) * m + j);
        }
    }
}

//  q(t) 
static void computeDischarge(const 매트릭스& I, const 매트릭스& T, const 매트릭스& FA, vector<double>& q) {
    int n = T.n, m = T.m;
    int maxT = 0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            if (T(i, j) > maxT) maxT = T(i, j);

    q.clear();
    if (maxT <= 0) return;

    vector<double> qk(maxT + 1, 0.0); // 1..maxT

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            int tt = T(i, j);
            if (tt >= 1 && tt <= maxT) qk[tt] += static_cast<double>(FA(i, j));
        }
    }

    q.reserve(static_cast<size_t>(maxT) + 2);
    q.push_back(0.0);
    q.push_back(static_cast<double>(I(n - 1, m - 1)));
    for (int k = 1; k <= maxT - 1; ++k) q.push_back(qk[k]);
    q.push_back(0.0);
}

//  QResult 
struct QResult
{
    매트릭스 FA;
    매트릭스 T;
    vector<double> q;
};

static QResult calculateQ2(const 매트릭스& D, const 매트릭스& I, int n0, int m0) {
    매트릭스 FA(D.n, D.m, 0);
    computeFlowAccum(D, I, FA);
    매트릭스 T(D.n, D.m, 0);
    도달시간계산(D, T, n0, m0);
    vector<double> q;
    computeDischarge(I, T, FA, q);
    return { FA, T, q };
}

//  loopcheck2: 도달성 
static bool loopcheck2_like_matlab(const 매트릭스& D, const 매트릭스& AD, int n0, int m0) {
    매트릭스 T(D.n, D.m, 0);
    도달시간계산(D, T, n0, m0);
    for (int i = 0; i < D.n; ++i) {
        for (int j = 0; j < D.m; ++j) {
            if (AD(i, j) != 0 && T(i, j) == 0) return true;
        }
    }
    return false;
}

//  changedirect2 
struct ChangeResult
{
    매트릭스 D;
    array<int, 4> r;
    int x = 0;
    int y = 0;
    bool ok = true;
};

static ChangeResult changeDirect2_matlab_like(
    const 매트릭스& D_in, const 매트릭스& AD, int n0, int m0, Xoshiro256ss& gen)
{
    int n = D_in.n, m = D_in.m;
    매트릭스 D = D_in;

    for (int tries = 0; tries < CHANGE_MAX_TRIES; ++tries)
    {
        int x = 0, y = 0;
        while (true)
        {
            x = gen.uniformInt(0, n - 1);
            y = gen.uniformInt(0, m - 1);
            if (AD(x, y) != 0) break;
        }
        if (x == n0 && y == m0) continue;

#if FIX_CELL_CONSTRAINT
        if (x == (FIX_I_1BASED - 1) && y == (FIX_J_1BASED - 1)) continue;
#endif

        int curdir = D(x, y);
        int newdir = gen.uniformInt(1, 4);
        if (newdir == curdir) continue;

        int xnew = x, ynew = y;
        if (newdir == 1) ynew = y + 1;
        else if (newdir == 2) xnew = x + 1;
        else if (newdir == 3) ynew = y - 1;
        else xnew = x - 1;

        if (xnew < 0 || xnew >= n || ynew < 0 || ynew >= m) continue;
        if (AD(xnew, ynew) == 0) continue;

        if (newdir == 1 && D(xnew, ynew) == 3) continue;
        if (newdir == 2 && D(xnew, ynew) == 4) continue;
        if (newdir == 3 && D(xnew, ynew) == 1) continue;
        if (newdir == 4 && D(xnew, ynew) == 2) continue;

        매트릭스 D_temp = D;
        D_temp(x, y) = newdir;
        if (loopcheck2_like_matlab(D_temp, AD, n0, m0)) continue;

        array<int, 4> r = { 1, 1, 1, 1 };
        if (curdir >= 1 && curdir <= 4) r[curdir - 1] -= 1;

        if (y + 1 >= m) r[0] -= 1;
        if (x - 1 < 0)  r[1] -= 1;
        if (y - 1 < 0)  r[2] -= 1;
        if (x + 1 >= n) r[3] -= 1;

        if (y - 1 >= 0 && D(x, y - 1) == 1) r[2] -= 1;
        if (x - 1 >= 0 && D(x - 1, y) == 2) r[1] -= 1;
        if (y + 1 < m && D(x, y + 1) == 3) r[0] -= 1;
        if (x + 1 < n && D(x + 1, y) == 4) r[3] -= 1;

        for (int k = 0; k < 4; ++k) if (r[k] < 0) r[k] = 0;

        D(x, y) = newdir;
        return { D, r, x, y, true };
    }

    // 실패
    return { D_in, {0,0,0,0}, 0, 0, false };
}

//  gibbs4 
static 매트릭스 gibbs4_cpp_matlab_like_seeded(
    int n, int m, double beta,
    const 매트릭스& AD, int n0, int m0,
    const 매트릭스& D1,
    uint64_t seed
) {
    매트릭스 D = D1;
    매트릭스 I(n, m, 1);
    Xoshiro256ss gen(seed);

    auto sumT = [](const 매트릭스& T) {
        long long s = 0;
        for (int v : T.data) s += v;
        return static_cast<double>(s);
        };

    int maxIter = 10 * n * m;

    for (int iter = 0; iter < maxIter; ++iter) {
        매트릭스 D_old = D;
        매트릭스 D_temp = D;

        auto changeRes = changeDirect2_matlab_like(D, AD, n0, m0, gen);
        if (!changeRes.ok) {
            // proposal 생성 자체가 실패 -> 이번 iter은 스킵(원복)
            D = D_temp;
            continue;
        }
        D = changeRes.D;
        const auto& r = changeRes.r;

        int rSum = r[0] + r[1] + r[2] + r[3];
        if (rSum <= 0) {
            D = D_temp;
            continue;
        }

        QResult resOld = calculateQ2(D_old, I, n0, m0);
        QResult resNew = calculateQ2(D, I, n0, m0);

        double H1 = sumT(resOld.T);
        double H2 = sumT(resNew.T);

        double expo = exp(-beta * (H2 - H1));
        double minpart = (expo < 1.0) ? expo : 1.0;
        double trans_pb12 = (1.0 / static_cast<double>(rSum)) * minpart;

        double u = gen.uniform01();
        if (u >= trans_pb12) {
            D = D_temp;
        }
    }

    return D;
}

//  파일 이름 변환 / 결과 쓰기 
static string makeResultFilename(const string& inputName)
{
    size_t pos = inputName.rfind('.');
    if (pos == string::npos) return inputName + "_결과.txt";
    return inputName.substr(0, pos) + "_결과" + inputName.substr(pos);
}

static void writeResultFileFromRes(const string& inputName, const 매트릭스& D, const QResult& res)
{
    string outName = makeResultFilename(inputName);
    ofstream ofs(outName);
    if (!ofs)
    {
        cerr << "결과 파일을 열 수 없음: " << outName << "\n";
        return;
    }
    int n = D.n, m = D.m;

    ofs << "//------ Final Direction Matrix D //------\n";
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j) ofs << D(i, j) << ' ';
        ofs << '\n';
    }

    ofs << "\n//------ Travel Time Matrix T //------\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) ofs << res.T(i, j) << ' ';
        ofs << '\n';
    }

    ofs << "\n//------ Flow Accumulation Matrix FA //------\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) ofs << res.FA(i, j) << ' ';
        ofs << '\n';
    }

    ofs << "\n//------ q(t) (index value) //------\n";
    for (size_t k = 0; k < res.q.size(); ++k) {
        ofs << k << ' ' << res.q[k] << '\n';
    }
}

//  폭 함수 분포 
static void computeWidthDistribution(const 매트릭스& D1, 매트릭스& FD, 매트릭스& LS) {
    int n = D1.n, m = D1.m;
    매트릭스 I(n, m, 1);
    FD = 매트릭스(n, m, 0);
    LS = 매트릭스(n, m, 0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            QResult res = calculateQ2(D1, I, i, j);
            double maxq = 0.0;
            for (double v : res.q) if (v > maxq) maxq = v;
            FD(i, j) = static_cast<int>(round(maxq));
            LS(i, j) = static_cast<int>(res.q.size()) - 1;
        }
    }
}

static vector<double> computeQVector(const 매트릭스& D1, int n0, int m0) {
    매트릭스 I(D1.n, D1.m, 1);
    QResult res = calculateQ2(D1, I, n0, m0);
    return res.q;
}

static void writeCsvMatrix(const 매트릭스& M, const string& filename) {
    ofstream ofs(filename);
    if (!ofs) {
        cerr << "CSV 파일을 열 수 없음: " << filename << "\n";
        return;
    }
    for (int i = 0; i < M.n; ++i) {
        for (int j = 0; j < M.m; ++j) {
            ofs << M(i, j);
            if (j < M.m - 1) ofs << ',';
        }
        ofs << '\n';
    }
}

static void writeWidthFunctionsCsv(
    const vector<string>& headers,
    const vector<vector<double>>& all_q,
    const string& filename
) {
    size_t maxLen = 0;
    for (const auto& v : all_q) if (v.size() > maxLen) maxLen = v.size();

    ofstream ofs(filename);
    if (!ofs) {
        cerr << "CSV 파일을 열 수 없음: " << filename << "\n";
        return;
    }

    ofs << "distance";
    for (const auto& h : headers) ofs << ',' << h;
    ofs << '\n';

    // 길이가 짧은 폭함수는 0으로 패딩해서 저장한다.
    for (size_t i = 0; i < maxLen; ++i) {
        ofs << i;
        for (size_t col = 0; col < all_q.size(); ++col) {
            double val = (i < all_q[col].size()) ? all_q[col][i] : 0.0;
            ofs << ',' << val;
        }
        ofs << '\n';
    }
}

//  NSE (Nash–Sutcliffe Efficiency)
static double 비교_NSE(const vector<double>& obs, const vector<double>& sim) {
    const size_t L = max(obs.size(), sim.size());
    if (L == 0) return numeric_limits<double>::quiet_NaN();

    double meanObs = 0.0;
    for (size_t i = 0; i < L; ++i) {
        double o = (i < obs.size()) ? obs[i] : 0.0;
        meanObs += o;
    }
    meanObs /= static_cast<double>(L);

    double denom = 0.0; // sum((obs - meanObs)^2)
    double numer = 0.0; // sum((obs - sim)^2)

    for (size_t i = 0; i < L; ++i) {
        double o = (i < obs.size()) ? obs[i] : 0.0;
        double s = (i < sim.size()) ? sim[i] : 0.0;

        double d = o - meanObs;
        denom += d * d;

        double e = o - s;
        numer += e * e;
    }

    // 관측(obs)이 완전 상수(분산 0)이면 NSE 분모가 0이 된다.
    // obs==sim이면 1로, 아니면 NaN으로 처리한다.
    if (denom <= 0.0) {
        if (numer <= 0.0) return 1.0;
        return numeric_limits<double>::quiet_NaN();
    }

    return 1.0 - (numer / denom);
}

static string makeNseFilename(const string& inputName) {
    // 예: myfile.txt -> myfile_NSE.txt
    size_t pos = inputName.rfind('.');
    if (pos == string::npos) return inputName + "_NSE.txt";
    return inputName.substr(0, pos) + "_NSE" + inputName.substr(pos);
}

//  병렬 배치: task 정의 
struct BatchTask {
    int k = 0;
    int run = 0;
    double beta = 0.0;
    size_t outIndex = 0; // all_q/headers에 넣을 인덱스
};

struct BatchResult {
    bool ok = true;
    string err;
    vector<double> q;
};

// 스레드 수 자동 + 상한
static int 스레드적용(int max_threads_user)
{
    unsigned hc = thread::hardware_concurrency();
    int t = (hc == 0) ? 4 : static_cast<int>(hc);
    if (max_threads_user > 0) t = min(t, max_threads_user);
    if (t < 1) t = 1;
    return t;
}

//  NSE 리포트 출력/저장 
// - 각 (k, run) 폭함수 q_sim vs 원본 q_original 의 NSE 계산
// - beta(k)별로 run 평균 폭함수(mean_q)를 만든 뒤, mean_q vs original NSE도 계산
static void reportNSE_andSave(
    const string& inputName,
    const vector<double>& q_original,
    const vector<BatchTask>& tasks,
    const vector<BatchResult>& results,
    const vector<vector<double>>& all_q,
    const vector<string>& headers
) {
    struct BetaAgg {
        int okRuns = 0;
        int totalRuns = 0;
        double sumNSE = 0.0;
        double minNSE = numeric_limits<double>::infinity();
        double maxNSE = -numeric_limits<double>::infinity();
        vector<double> sumQ; // run 평균 폭함수 계산용 누적합(부족한 tail은 0으로 간주)
    };

    map<int, BetaAgg> agg; // key = k
    vector<double> nse_run(tasks.size(), numeric_limits<double>::quiet_NaN());

    // 1) run별 NSE 계산 + beta별 누적(요약용)
    for (size_t ti = 0; ti < tasks.size(); ++ti) {
        const auto& task = tasks[ti];
        agg[task.k].totalRuns++;

        // 실패 task는 all_q[task.outIndex]가 빈 벡터로 들어가 있음(메인에서 그렇게 처리)
        const bool ok = results[ti].ok && !all_q[task.outIndex].empty();
        if (!ok) continue;

        const vector<double>& q_sim = all_q[task.outIndex];
        double nse = 비교_NSE(q_original, q_sim);
        nse_run[ti] = nse;

        BetaAgg& A = agg[task.k];
        A.okRuns++;

        if (!std::isnan(nse)) {
            A.sumNSE += nse;
            A.minNSE = min(A.minNSE, nse);
            A.maxNSE = max(A.maxNSE, nse);
        }

        // 평균 폭함수(mean_q) 만들기 위해 누적합 계산
        if (A.sumQ.size() < q_sim.size()) A.sumQ.resize(q_sim.size(), 0.0);
        for (size_t i = 0; i < q_sim.size(); ++i) A.sumQ[i] += q_sim[i];
    }

    // 2) 파일 오픈
    string nseFile = makeNseFilename(inputName);
    ofstream nfs(nseFile);
    if (!nfs) {
        cerr << "NSE 결과 파일을 열 수 없음: " << nseFile << "\n";
    }

    // 3) 콘솔/파일 출력
    cout << "\n\n//------ NSE 비교 결과 (원본 폭함수 vs beta별 폭함수) //------\n";
    cout << "규칙: 길이 다르면 0으로 패딩 후 NSE 계산\n";

    if (nfs) {
        nfs << "//------ NSE 비교 결과 (원본 폭함수 vs beta별 폭함수) //------\n";
        nfs << "규칙: 길이 다르면 0으로 패딩 후 NSE 계산\n\n";
    }

    // ----- run별 상세 -----
    cout << "\n[run별 NSE]\n";
    if (nfs) {
        nfs << "[run별 NSE]\n";
        nfs << "k\tbeta\trun\tok\tNSE\tlabel\n";
    }

    cout << fixed << setprecision(6);
    for (size_t ti = 0; ti < tasks.size(); ++ti) {
        const auto& task = tasks[ti];
        const bool ok = results[ti].ok && !all_q[task.outIndex].empty();
        const double nse = nse_run[ti];

        cout << "k=" << task.k
            << "  beta=" << scientific << setprecision(3) << task.beta
            << fixed << setprecision(6)
            << "  run=" << task.run
            << "  ok=" << (ok ? 1 : 0)
            << "  NSE=";
        if (std::isnan(nse)) cout << "NaN";
        else cout << nse;
        cout << "\n";

        if (nfs) {
            nfs << task.k << "\t"
                << scientific << setprecision(10) << task.beta
                << fixed << setprecision(10) << "\t"
                << task.run << "\t"
                << (ok ? 1 : 0) << "\t";
            if (std::isnan(nse)) nfs << "NaN\t";
            else nfs << nse << "\t";
            nfs << headers[task.outIndex] << "\n";
        }
    }

    // ----- beta(k)별 요약 -----
    cout << "\n[beta별 요약]\n";
    cout << "k\t\tbeta\t\tok/total\t\tmeanNSE(run)\t\tminNSE\t\tmaxNSE\t\tNSE(mean_q)\n";

    if (nfs) {
        nfs << "\n[beta별 요약]\n";
        nfs << "k\t\tbeta\t\tok/total\t\tmeanNSE(run)\t\tminNSE\t\tmaxNSE\t\tNSE(mean_q)\n";
    }

    cout << fixed << setprecision(6);
    for (const auto& kv : agg) {
        const int k = kv.first;
        const BetaAgg& A = kv.second;
        const double betaVal = pow(10.0, k);

        double meanNSE_run = numeric_limits<double>::quiet_NaN();
        if (A.okRuns > 0) meanNSE_run = A.sumNSE / static_cast<double>(A.okRuns);

        // beta별 run 평균 폭함수(mean_q) 구성
        vector<double> meanQ = A.sumQ;
        if (A.okRuns > 0) {
            for (double& v : meanQ) v /= static_cast<double>(A.okRuns);
        }
        else {
            meanQ.clear();
        }
        double nse_meanQ = 비교_NSE(q_original, meanQ);

        // 콘솔
        cout << k << "\t"
            << scientific << setprecision(3) << betaVal
            << fixed << setprecision(6) << "\t"
            << A.okRuns << "/" << A.totalRuns << "\t\t";

        if (std::isnan(meanNSE_run)) cout << "NaN\t\t";
        else cout << meanNSE_run << "\t\t";

        if (std::isinf(A.minNSE)) cout << "NaN\t";
        else cout << A.minNSE << "\t";

        if (std::isinf(A.maxNSE)) cout << "NaN\t";
        else cout << A.maxNSE << "\t";

        if (std::isnan(nse_meanQ)) cout << "NaN\n";
        else cout << nse_meanQ << "\n";

        // 파일(좀 더 높은 정밀도)
        if (nfs) {
            nfs << k << "\t"
                << scientific << setprecision(10) << betaVal
                << fixed << setprecision(10) << "\t"
                << A.okRuns << "/" << A.totalRuns << "\t";

            if (std::isnan(meanNSE_run)) nfs << "NaN\t";
            else nfs << meanNSE_run << "\t";

            if (std::isinf(A.minNSE)) nfs << "NaN\t";
            else nfs << A.minNSE << "\t";

            if (std::isinf(A.maxNSE)) nfs << "NaN\t";
            else nfs << A.maxNSE << "\t";

            if (std::isnan(nse_meanQ)) nfs << "NaN\n";
            else nfs << nse_meanQ << "\n";
        }
    }

    if (nfs) {
        nfs << "\n(끝)\n";
        nfs.close();
    }

    cout << "\nNSE 저장: " << nseFile << "\n";
}

void Consol_setting()
{
    system("mode con cols=150 lines=150");
}

//  main 
int main(int argc, char* argv[])
{
//    void Consol_setting();
    try {
        string inputName;
        if (argc > 1) inputName = argv[1];
        else {
            inputName = chooseTxtFileInteractive();
            if (inputName.empty()) return 0;
        }

        ifstream ifs(inputName);
        if (!ifs) {
            cerr << "입력 파일을 열 수 없음: " << inputName << "\n";
            cerr << "CWD = " << fs::current_path().string() << "\n";
            cout << "엔터를 누르면 종료...\n";
            cin.get();
            cin.get();
            return 1;
        }

        int n, m;
        if (!(ifs >> n >> m)) {
            cerr << "파일 형식 오류: n m 읽기 실패\n";
            return 1;
        }

        int n0_1, m0_1;
        if (!(ifs >> n0_1 >> m0_1)) {
            cerr << "파일 형식 오류: outlet(n0 m0) 읽기 실패\n";
            return 1;
        }

        int n0 = n0_1 - 1;
        int m0 = m0_1 - 1;
        if (n <= 0 || m <= 0 || n0 < 0 || n0 >= n || m0 < 0 || m0 >= m) {
            cerr << "입력값 범위 오류\n";
            return 1;
        }

        매트릭스 D1(n, m, 0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                int val;
                if (!(ifs >> val)) {
                    cerr << "파일 형식 오류: D 행렬 데이터 부족\n";
                    return 1;
                }
                D1(i, j) = val;
            }
        }
        ifs.close();

        매트릭스 AD(n, m, 0);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j)
                AD(i, j) = (D1(i, j) != 0) ? 1 : 0;

        if (AD(n0, m0) == 0) {
            cerr << "오류: outlet 위치가 AD(유효 셀) 밖입니다. (D1이 0)\n";
            return 1;
        }

        // //------== beta 입력 //------==
        double a;
        cout << "\n베타를 beta = 10^a 로 설정합니다.\n";
        cout << "a 값을 입력하세요(예: 6 -> 1e6): ";
        if (!(cin >> a)) {
            cerr << "a 입력 실패(콘솔 입력이 없으면 더블클릭 실행이 아닌 터미널 실행 권장)\n";
            return 1;
        }
        if (a > 308.0 || a < -308.0) {
            cerr << "a 범위 오류: -308 ~ 308\n";
            return 1;
        }
        double beta = pow(10.0, a);
        cout << "설정된 beta = " << beta << " (10^" << a << ")\n";

        // //------== Gibbs 단일 실행 //------==
        uint64_t baseSeed = static_cast<uint64_t>(
            chrono::high_resolution_clock::now().time_since_epoch().count()
            );
        auto t0 = tic();
        매트릭스 Dgibbs = gibbs4_cpp_matlab_like_seeded(
            n, m, beta, AD, n0, m0, D1, mix_seed(baseSeed ^ 0xA5A5A5A5ULL)
        );
        매트릭스 I(n, m, 1);
        QResult finalRes = calculateQ2(Dgibbs, I, n0, m0);
        double elapsed = toc_sec(t0);

        writeResultFileFromRes(inputName, Dgibbs, finalRes);

        cout << "\n[계산 시간] " << elapsed << " sec"
            << "  (txt 로드+beta 입력 이후 ~ 계산 종료, 파일 저장시간 제외)\n";
        cout << "완료: " << makeResultFilename(inputName) << "\n";

        // //------== 폭 함수(확장 기능) //------==
        char doWidth;
        cout << "\n폭 함수(width function) 계산을 수행하시겠습니까? (y/n): ";
        cin >> doWidth;

        if (doWidth == 'y' || doWidth == 'Y') {
            매트릭스 FD, LS;
            cout << "폭 함수 분포(FD, LS)를 계산 중입니다...\n";
            computeWidthDistribution(D1, FD, LS);

            string baseName;
            {
                size_t pos = inputName.find_last_of("/\\");
                baseName = (pos == string::npos) ? inputName : inputName.substr(pos + 1);
            }

            string fdName = baseName + "_FD.csv";
            string lsName = baseName + "_LS.csv";
            writeCsvMatrix(FD, fdName);
            writeCsvMatrix(LS, lsName);
            cout << "저장: " << fdName << ", " << lsName << "\n";

            // 원본 q + 배치 q
            vector<vector<double>> all_q;
            vector<string> headers;

            // all_q[0] = 원본 폭함수(원본 D1 기준 outlet q)
            all_q.push_back(computeQVector(D1, n0, m0));
            headers.push_back("original");

            char doBatch;
            cout << "여러 beta(10^k)으로 폭함수를 저장할까요? (y/n): ";
            cin >> doBatch;

            if (doBatch == 'y' || doBatch == 'Y') {
                vector<int> k_vals = { -4, -3, -2, -1, 0, 1, 2, 3 };
                int runsPerBeta = 100; // 기본 100회
                char customRange;
                cout << "기본값: k=-4~3, run=100. 변경?(y/n): ";
                cin >> customRange;
                if (customRange == 'y' || customRange == 'Y') {
                    int kStart, kEnd;
                    cout << "k 시작(정수): "; cin >> kStart;
                    cout << "k 끝(정수): ";   cin >> kEnd;
                    if (kStart > kEnd) swap(kStart, kEnd);
                    k_vals.clear();
                    for (int k = kStart; k <= kEnd; ++k) k_vals.push_back(k);
                    cout << "반복(run) 횟수: "; cin >> runsPerBeta;
                    if (runsPerBeta <= 0) runsPerBeta = 1;
                }

                int max_threads_user = 0;
                cout << "스레드 상한(max_threads). 0이면 자동(hardware_concurrency): ";
                cin >> max_threads_user;

                int nThreads = 스레드적용(max_threads_user);
                cout << "병렬 배치 실행: threads = " << nThreads
                    << "  (hc=" << thread::hardware_concurrency() << ")\n";

                // ---- task 생성 ----
                vector<BatchTask> tasks;
                tasks.reserve(static_cast<size_t>(k_vals.size()) * static_cast<size_t>(runsPerBeta));

                // 결과 저장 공간: all_q / headers는 push_back 금지(병렬) -> 미리 크기 확장 후 인덱스 저장
                size_t baseIndex = all_q.size(); // original 다음부터
                size_t totalTasks = static_cast<size_t>(k_vals.size()) * static_cast<size_t>(runsPerBeta);

                all_q.resize(baseIndex + totalTasks);
                headers.resize(baseIndex + totalTasks);

                size_t idx = baseIndex;
                for (int k : k_vals) {
                    double bval = pow(10.0, k);
                    for (int run = 1; run <= runsPerBeta; ++run) {
                        BatchTask t;
                        t.k = k;
                        t.run = run;
                        t.beta = bval;
                        t.outIndex = idx++;
                        tasks.push_back(t);

                        headers[t.outIndex] = "10^" + to_string(k) + "_run" + to_string(run);
                    }
                }

                vector<BatchResult> results(tasks.size());
                atomic<size_t> nextTask{ 0 };
                atomic<size_t> doneCount{ 0 };
                mutex ioMutex;

                auto worker = [&](int tid) {
                    while (true) {
                        size_t tIndex = nextTask.fetch_add(1);
                        if (tIndex >= tasks.size()) break;
                        const auto& task = tasks[tIndex];

                        try {
                            uint64_t seed = mix_seed(baseSeed ^ (0x9E3779B97f4A7C15ULL * (tIndex + 1)) ^ (uint64_t)tid);
                            매트릭스 Dg = gibbs4_cpp_matlab_like_seeded(n, m, task.beta, AD, n0, m0, D1, seed);
                            results[tIndex].q = computeQVector(Dg, n0, m0);
                            results[tIndex].ok = true;
                        }
                        catch (const exception& e) {
                            results[tIndex].ok = false;
                            results[tIndex].err = e.what();
                        }
                        catch (...) {
                            results[tIndex].ok = false;
                            results[tIndex].err = "unknown error";
                        }

                        size_t d = doneCount.fetch_add(1) + 1;
                        if ((d % 10) == 0 || d == tasks.size()) {
                            lock_guard<mutex> lk(ioMutex);
                            cout << "\r계산상황: " << d << " / " << tasks.size() << flush;
                        }
                    }
                    };

                auto tb0 = tic();
                vector<thread> pool;
                pool.reserve(nThreads);
                for (int t = 0; t < nThreads; ++t) pool.emplace_back(worker, t);
                for (auto& th : pool) th.join();
                double batchSec = toc_sec(tb0);
                cout << "\n배치 완료. batch time = " << batchSec << " sec\n";

                // ---- 결과를 all_q에 반영 (순차) ----
                size_t failCount = 0;
                for (size_t ti = 0; ti < tasks.size(); ++ti) {
                    const auto& task = tasks[ti];
                    if (!results[ti].ok) {
                        ++failCount;
                        // 실패 task는 빈 벡터 -> CSV 저장 시 0으로 패딩됨
                        all_q[task.outIndex] = {};
                    }
                    else {
                        all_q[task.outIndex] = move(results[ti].q);
                    }
                }
                if (failCount > 0) {
                    cout << "경고: 실패 task = " << failCount << " / " << tasks.size()
                        << " (입력/제약/무한루프 방지로 proposal 실패 등 가능)\n";
                }

                // //------== NSE 비교 + 콘솔 출력 + txt 저장 //------==
                // 각 beta(run 포함) 폭함수 vs 원본 폭함수 NSE 비교 결과를 파일로도 남긴다.
                reportNSE_andSave(inputName, all_q[0], tasks, results, all_q, headers);
            }

            string wfname = fs::path(inputName).filename().string() + "_width_functions.csv";
            writeWidthFunctionsCsv(headers, all_q, wfname);
            cout << "저장: " << wfname << "\n";
        }

        cout << "\n끝~\n아무거나 입력하고 엔터하면 종료.MADE BY 영남대 건시공 스마트 수자원\n";
        int dummy;
        if (!(cin >> dummy)) {
            // 더블클릭 실행 등 stdin 종료 상황 대비
            cin.clear();
            cout << "(입력 스트림이 닫혀있음) 엔터로 종료...\n";
            cin.get();
        }
        return 0;

    }
    catch (const exception& e) {
        cerr << "\n[치명적 예외] " << e.what() << "\n";
        cout << "엔터를 누르면 종료...\n";
        cin.get();
        cin.get();
        return 1;
    }
    catch (...) {
        cerr << "\n[치명적 예외] unknown\n";
        cout << "엔터를 누르면 종료...\n";
        cin.get();
        cin.get();
        return 1;
    }
}
