.PHONY: all build run

all: build run

build: clean
	g++ main.cpp -O3 -Wall -I./raylib-5.5_linux_amd64/include/ -o ./build/main -L./raylib-5.5_linux_amd64/lib/ -l:libraylib.a -lm

run:
	./build/main

clean:
	rm -rf ./build
	mkdir -p ./build
