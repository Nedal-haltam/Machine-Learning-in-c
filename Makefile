



all: build run

build: Neuralizer.cpp
	g++ Neuralizer.cpp -Wall -Wextra -Wpedantic -Iraylib/include -Lraylib/lib -lraylib -lgdi32 -lwinmm -std=c++20 -o out.exe

run: out.exe
	.\out.exe
