#include <stdio.h>

int main() {
    const char *str0 = "Hello";
    const char *str = str0;  // str is a pointer to a constant string

    printf("%s\n", *str);

    // str[0] = 'h';  // ❌ ERROR: You cannot modify the string because it's `const`
}

