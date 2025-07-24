#!/bin/bash

echo "Compiling programs..."
make

if [ ! -f "exe_fs_main" ] || [ ! -f "exe_fs_serial" ]; then
    echo "Compilation failed!"
    echo "Expected files: exe_fs_main, exe_fs_serial"
    exit 1
fi

TESTCASE_DIR="testcases"

if [ ! -d "$TESTCASE_DIR" ]; then
    echo "Error: testcases directory not found!"
    exit 1
fi

OUTPUT_DIR="temp_outputs"
mkdir -p $OUTPUT_DIR

total_tests=0
passed_tests=0

echo "Running tests from $TESTCASE_DIR..."

for testfile in $TESTCASE_DIR/*.in; do
    if [ -f "$testfile" ]; then
        testname=$(basename "$testfile" .in)
        total_tests=$((total_tests + 1))
        
        echo "Testing case: $testname..."
        
        ./exe_fs_main "$testfile" > $OUTPUT_DIR/gpu_output_$testname.txt 2>/dev/null
        gpu_status=$?
        
        ./exe_fs_serial "$testfile" > $OUTPUT_DIR/cpu_output_$testname.txt 2>/dev/null
        cpu_status=$?
        
        if [ $gpu_status -ne 0 ] || [ $cpu_status -ne 0 ]; then
            echo "✗ Test $testname: Program execution failed!"
            if [ $gpu_status -ne 0 ]; then
                echo "  GPU version failed with exit code: $gpu_status"
            fi
            if [ $cpu_status -ne 0 ]; then
                echo "  CPU version failed with exit code: $cpu_status"
            fi
        else
            if diff -q $OUTPUT_DIR/gpu_output_$testname.txt $OUTPUT_DIR/cpu_output_$testname.txt > /dev/null; then
                echo "✓ Test $testname: Results match!"
                passed_tests=$((passed_tests + 1))
            else
                echo "✗ Test $testname: Results differ!"
            fi
        fi
    fi
done

echo "========================================="
echo "Test Summary"
echo "========================================="
echo "Total tests: $total_tests"
echo "Passed tests: $passed_tests"
echo "Failed tests: $((total_tests - passed_tests))"

if [ $passed_tests -eq $total_tests ] && [ $total_tests -gt 0 ]; then
    echo "🎉 All tests passed!"
    exit_code=0
else
    echo "❌ Some tests failed!"
    exit_code=1
fi

echo "Cleaning up temporary files..."
rm -rf $OUTPUT_DIR

exit $exit_code