# Clang Compilation Steps Guide

Understanding the different stages of compilation with clang/clang++ and the files they produce.

## Important Notes

- Use `clang++` for C++ files (`.cc`, `.cpp`, `.cxx`)
- Use `clang` for C files (`.c`)
- Preprocessed files need the same compiler as the original source



## Step 1: Preprocessing (`-E`)

```bash
clang++ -E test.cc > preprocessed.i
```

**What it does:** Runs only the preprocessor - expands `#include` directives, processes `#define` macros, handles conditional compilation (`#ifdef`), and removes comments.

**File produced:** `preprocessed.i`
- **Content:** Pure C/C++ code with all headers included and macros expanded
- **Purpose:** Shows exactly what the compiler sees after preprocessing; useful for debugging macro issues

## Step 2: Compilation to Assembly (`-S`)

```bash
clang++ -S test.cc
# or from preprocessed file
clang++ -S preprocessed.i
```


**What it does:** Compiles the source code to assembly language but doesn't assemble it into machine code.

**File produced:** `test.s` (or `preprocessed.s` if using preprocessed input)
- **Content:** Human-readable assembly language instructions for the target architecture
- **Purpose:** Shows the actual assembly instructions generated; useful for performance analysis and understanding compiler optimizations

## Step 3: Assembly to Object Code (`-c`)

```bash
clang++ -c test.cc -o test.o
# or from assembly file
clang++ -c test.s -o test.o
```


**What it does:** Assembles the code into machine code but doesn't link it with libraries or other object files.

**File produced:** `test.o` (object file)
- **Content:** Binary machine code with unresolved external references and symbol table
- **Purpose:** Relocatable machine code that can be linked with other object files and libraries to create an executable

## Step 4: Linking (default behavior)

```bash
clang++ test.cc -o executable
# or from object file
clang++ test.o -o executable
```

**What it does:** Links object files with required libraries and resolves all symbol references to create the final executable.

**File produced:** `executable` (or specified name)
- **Content:** Complete executable binary with all dependencies resolved
- **Purpose:** Final runnable program that can be executed directly

## Additional Options

### Preserve Comments (`-C` with `-E`)
```bash
clang++ -E -C test.cc > preprocessed_with_comments.i
clang -E -C test.c > preprocessed_with_comments.i
```
- Keeps comments in the preprocessed output (only works with `-E`)

## Complete Build Process Examples

### C++ Step-by-Step:
```bash
clang++ -E test.cc > preprocessed.i        # Preprocess
clang++ -S preprocessed.i                  # Compile to assembly (produces preprocessed.s)
clang++ -c preprocessed.s -o test.o        # Assemble to object code
clang++ test.o -o final_executable         # Link to create executable
./final_executable                         # Run the program
```

### C Step-by-Step:
```bash
clang -E test.c > preprocessed.i           # Preprocess
clang -S preprocessed.i                    # Compile to assembly (produces preprocessed.s)
clang -c preprocessed.s -o test.o          # Assemble to object code
clang test.o -o final_executable           # Link to create executable
./final_executable                         # Run the program
```

### All-in-One Commands:
```bash
# C++
clang++ test.cc -o final_executable

# C
clang test.c -o final_executable
```


## File Extensions Summary

| Extension | Content | Stage | Compiler |
|-----------|---------|-------|----------|
| `.c` | C source code | Input | `clang` |
| `.cc/.cpp/.cxx` | C++ source code | Input | `clang++` |
| `.i/.ii` | Preprocessed source | After preprocessing | Same as input |
| `.s` | Assembly language | After compilation | Same as input |
| `.o` | Object code (binary) | After assembly | Same as input |
| (no ext) | Executable binary | After linking | Same as input |

## Sample Output

```
Hi 10 there! 20 Hello 30
```