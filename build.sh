
clang -O0 -g -Wall -Wextra -std=c11 msdf.c -c -o libmsdf.o
odin build . -o:none -debug

