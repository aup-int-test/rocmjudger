// elemcheck.cpp
// Element-wise double-precision comparator for ref vs out.
// Usage:
//   ./elemcheck ref.txt out.txt [--atol 1e-8] [--rtol 1e-6] [--mind 1e-12] [--show 5]
//   ./elemcheck ref.txt out.txt --atol 1e-8 --rtol 1e-6 --mind 1e-12 --show 5
// Notes:
// - Each file may be either: "N" followed by N numbers, OR just N numbers without a header.
// - PASS if for every i: abs(out[i]-ref[i]) <= atol  OR  rel <= rtol,
//   where rel = abs/ max(|ref[i]|, mind).
// Exit codes: 0=PASS, 1=FAIL, 2=ERROR

#include <bits/stdc++.h>
using namespace std;

static bool is_integer_string(const string& s) {
    if (s.empty()) return false;
    size_t i = 0;
    if (s[0] == '+' || s[0] == '-') i = 1;
    if (i >= s.size()) return false;
    for (; i < s.size(); ++i) if (!isdigit(static_cast<unsigned char>(s[i]))) return false;
    return true;
}

static vector<double> load_vec(const string& path) {
    ifstream f(path);
    if (!f) throw runtime_error("Cannot open file: " + path);
    vector<string> toks;
    toks.reserve(1<<20);
    string s;
    while (f >> s) toks.push_back(s);
    if (toks.empty()) throw runtime_error("Empty file: " + path);

    if (is_integer_string(toks[0])) {
        long long N = -1;
        try { N = stoll(toks[0]); } catch (...) { N = -1; }
        if (N >= 0 && static_cast<size_t>(N) == toks.size() - 1) {
            vector<double> v(N);
            for (long long i = 0; i < N; ++i) {
                try { v[i] = stod(toks[1 + i]); }
                catch (...) { throw runtime_error("Bad number at index " + to_string(i) + " in " + path); }
            }
            return v;
        }
    }
    vector<double> v(toks.size());
    for (size_t i = 0; i < toks.size(); ++i) {
        try { v[i] = stod(toks[i]); }
        catch (...) { throw runtime_error("Bad number token at position " + to_string(i) + " in " + path + ": '" + toks[i] + "'"); }
    }
    return v;
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " REF OUT [--atol 1e-8] [--rtol 1e-6] [--mind 1e-12] [--show 5]\n";
        return 2;
    }
    string ref_path = argv[1], out_path = argv[2];
    double atol = 1e-8, rtol = 1e-6, mind = 1e-12;
    int show = 5;

    for (int i = 3; i < argc; ++i) {
        string a = argv[i];
        auto getd = [&](double& dst){ if (i+1>=argc) { cerr<<"Missing value after "<<a<<"\n"; exit(2);} dst = stod(argv[++i]); };
        auto geti = [&](int& dst){ if (i+1>=argc) { cerr<<"Missing value after "<<a<<"\n"; exit(2);} dst = stoi(argv[++i]); };
        if (a=="--atol") getd(atol);
        else if (a=="--rtol") getd(rtol);
        else if (a=="--mind") getd(mind);
        else if (a=="--show") geti(show);
        else { cerr<<"Unknown arg: "<<a<<"\n"; return 2; }
    }

    vector<double> ref, out;
    try { ref = load_vec(ref_path); out = load_vec(out_path); }
    catch (const exception& e) { cerr << "ERROR: " << e.what() << "\n"; return 2; }

    if (ref.size() != out.size()) {
        cerr << "Shape mismatch: ref " << ref.size() << " vs out " << out.size() << "\n";
        return 2;
    }

    size_t N = ref.size(), mism = 0;
    double max_abs = 0.0, max_rel = 0.0;
    size_t max_abs_idx = 0, max_rel_idx = 0;
    vector<size_t> first_bad;
    first_bad.reserve((size_t)show);

    for (size_t i = 0; i < N; ++i) {
        double r = ref[i], o = out[i];
        double ad = fabs(o - r);
        double denom = max(fabs(r), mind);
        double rd = ad / denom;
        if (ad > max_abs) { max_abs = ad; max_abs_idx = i; }
        if (rd > max_rel) { max_rel = rd; max_rel_idx = i; }
        bool ok = (ad <= atol) || (rd <= rtol);
        if (!ok) {
            if (first_bad.size() < (size_t)show) first_bad.push_back(i);
            ++mism;
        }
    }

    cout.setf(std::ios::scientific);
    cout << setprecision(12);

    if (mism == 0) {
        cout << "PASS ✅  (N=" << N << ", max_abs=" << max_abs << ", max_rel=" << max_rel << ")\n";
        return 0;
    } else {
        cout << "FAIL ❌\n";
        cout << "  N=" << N << ", mismatches=" << mism << "\n";
        cout << "  max_abs=" << max_abs << " at idx " << max_abs_idx
             << " (ref=" << ref[max_abs_idx] << ", out=" << out[max_abs_idx] << ")\n";
        cout << "  max_rel=" << max_rel << " at idx " << max_rel_idx
             << " (ref=" << ref[max_rel_idx] << ", out=" << out[max_rel_idx] << ")\n";
        for (size_t k = 0; k < first_bad.size(); ++k) {
            size_t i = first_bad[k];
            double ad = fabs(out[i] - ref[i]);
            double rd = ad / max(fabs(ref[i]), mind);
            cout << "  idx " << i << ": ref=" << ref[i] << " out=" << out[i]
                 << " abs=" << ad << " rel=" << rd << "\n";
        }
        return 1;
    }
}
