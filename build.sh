#!/bin/sh

set -xe

# gcc -Wall -Wextra -o build/twice twice.c -lm
# gcc -Wall -Wextra -o build/main main.c -lm
gcc -Wall -Wextra -o build/xor xor.c -lm
gcc -Wall -Wextra -o build/adder adder.c -lm

