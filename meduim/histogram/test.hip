#include <iostream>
#include <cstdlib>
#include <string>

int main() {
    system("make");
    
    std::string executables[] = {"exe_fs_serial", "exe_fs_bad", "exe_fs_main"};
    
    for (int exe = 0; exe < 3; exe++) {
        for (int testcase = 1; testcase <= 6; testcase++) {
            std::string command = "./" + executables[exe] + " testcases/" + std::to_string(testcase) + ".in > tmp";
            system(command.c_str());
        }
    }
    
    system("rm tmp");
    
    return 0;
}