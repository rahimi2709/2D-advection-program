# 2D-advection-program
Outputs: initial.dat - inital values of u(x,y)          final.dat   - final values of u(x,y) &amp; VerticalAVG.dat - final values of vertically averaged distribution of u(x,y)           The {final.dat} have three columns: x, y, u          The {VerticalAVG.dat} have two columns: x, vert_avg(vertically averaged of u)  For series method compile with:  gcc -o Source_Code_RahimiFard -std=c99 Source_Code_RahimiFard.c -lm For series method compile with:  gcc -fopenmp -o Source_Code_RahimiFard -std=c99 Source_Code_RahimiFard.c -lm
