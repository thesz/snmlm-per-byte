all: deco

deco: deco.c
	gcc -o deco -O3 -g -ffast-math -mavx -mavx2 deco.c -lm
deco-d: deco.c
	gcc -o deco-d -g deco.c -lm

