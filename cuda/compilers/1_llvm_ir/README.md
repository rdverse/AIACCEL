# LLVM IR and Pass Development Guide

## Generating LLVM IR and Assembly

### LLVM IR Generation
```bash
# Generate readable LLVM IR
clang -emit-llvm -S -o llvm_ir.out hello.c
cat llvm_ir.out
```

### Assembly Generation
```bash
# Generate assembly with clang
clang -S -o llvm_asm.out hello.c
cat llvm_asm.out

# Generate assembly with g++
g++ -g -S hello.c -o hello.s
```

### GIMPLE Tree Dump
```bash
# Generate GIMPLE tree representation
g++ -fdump-tree- -o gimple.out hello.c
```
Note: GIMPLE is GCC's intermediate representation, similar to LLVM IR.

## LLVM Pass Development

### Building the Pass
```bash
# Clean and rebuild the pass
rm a.out
cd build
cmake ..
make
cd ..
```

### Running the Pass
```bash
# Run the pass on your code
clang -fpass-plugin=`echo build/skeleton/SkeletonPass.*` hello.c
./a.out 5 8
```

### Adding Print Log Statements
```bash
# Compile with logging support
cc -c log.c
clang -Xclang -load -Xclang build/skeleton/SkeletonPass.so -c hello.c
cc log.o hello.o -o exe
./exe 10 20
```

## Notes
- LLVM IR (Intermediate Representation) is a low-level programming language similar to assembly
- The pass system allows you to analyze and transform LLVM IR
- Use `-S` flag to generate human-readable output
- Use `-emit-llvm` to generate LLVM IR instead of machine code






