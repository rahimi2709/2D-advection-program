*********************************************************************
                     Include header files
**********************************************************************/

#include <stdio.h>
#include <math.h>
#ifdef _OPENMP /*  To use in Parallel Mode by OPENMP */
#include <omp.h>
#endif

/*********************************************************************
                      Main function
**********************************************************************/

int main(){

  /* Grid properties */
  const int NX=1000;     // Number of x points
  const int NY=1000;     // Number of y points
  const float xmin=0.0;  // Minimum x value
  const float xmax=30.0; // Maximum x value
  const float ymin=0.0;  // Minimum y value
  const float ymax=30.0; // Maximum y value

  /* Parameters for the Gaussian initial conditions */
  const float x0=0.1;                     // Centre(x)
  const float y0=15.0;                    // Centre(y)
  const float sigmax=0.03;                // Width(x)
  const float sigmay=5.0;                 // Width(y)
  const float sigmax2 = sigmax * sigmax;  // Width(x) squared
  const float sigmay2 = sigmay * sigmay;  // Width(y) squared

  /* Boundary conditions */
  const float bval_left=0.0;    // Left boundary value
  const float bval_right=0.0;   // Right boundary value
  const float bval_lower=0.0;   // Lower boundary
  const float bval_upper=0.0;   // Upper boundary

  /* Time stepping parameters */
  const float CFL=0.9;   // CFL number
  const int nsteps=1000; // Number of time steps

  /* Velocity */
  /* velocity in x direction is not constant so it is removed from here but VxMAX is calculated in Section 2.3*/
  const float vely=0.0; // Velocity in y direction

  /* Arrays to store variables. These have NX+2 elements
     to allow boundary values to be stored at both ends */
  float x[NX+2];          // x-axis values
  float y[NX+2];          // y-axis values
  float u[NX+2][NY+2];    // Array of u values
  float dudt[NX+2][NY+2]; // Rate of change of u
  /* float x2;   // x squared (initial conditions=0 so it will not used) /*
  /* float y2;   // y squared (initial conditions=0 so it will not used) /*

  /* Calculate distance between points */
  float dx = (xmax-xmin) / ( (float) NX);
  float dy = (ymax-ymin) / ( (float) NY);

  /* ###################Added New Parameters: ############################### */

  /* Parameters for section 2.2 */
  const float sigmat = 1.0;
  const float sigmat2 = sigmat * sigmat;
  const float t0 = 3.0;

  /* Parameters for section 2.3 */
  float VxVAR ; // For calculating variable horizontal speed in Formula (1)
  const float k = 0.41;
  const float z0 = 1.0;
  const float u_star = 0.1;
  const float VxMAX=(u_star/k) * log(ymax/z0); // Considering elevation y=30.0m in Formula (1) to find maximum velocity in
                                           // x direction and then using for calculating dt.

  /* Parameters for section 2.4 */
  float vert_avg[NY];
  float temp_sum=0;

  /* Calculate time step using the CFL condition */
  /* The fabs function gives the absolute value in case the velocity is -ve */
  float dt = CFL / ( (fabs(VxMAX) / dx) + (fabs(vely) / dy) );

  /*** Report information about the calculation ***/
  printf("Grid spacing dx     = %g\n", dx);
  printf("Grid spacing dy     = %g\n", dy);
  printf("CFL number          = %g\n", CFL);
  printf("Time step           = %g\n", dt);
  printf("No. of time steps   = %d\n", nsteps);
  printf("End time            = %g\n", dt*(float) nsteps);
/*printf("Distance advected x = %g\n", velx*dt*(float) nsteps); >> Because the Horizontal Velocity is variable so the calculated
                                                                   Distance Advected is not valid and commented out from code*/
  printf("Distance advected y = %g\n", vely*dt*(float) nsteps);

  /*** Place x points in the middle of the cell ***/
  /* LOOP 1 >> parallelized*/
       #ifdef _OPENMP
        #pragma omp parallel for default (none) shared(dx,x)
           for (int i=0; i<NX+2; i++){
              x[i] = ( (float) i - 0.5) * dx;
            }
        #else
            for (int i=0; i<NX+2; i++){
              x[i] = ( (float) i - 0.5) * dx;
            }
        #endif


  /*** Place y points in the middle of the cell ***/
  /* LOOP 2 >> parallelized*/
      #ifdef _OPENMP
        #pragma omp parallel for default (none) shared(dy,y)
        for (int j=0; j<NY+2; j++){
          y[j] = ( (float) j - 0.5) * dy;
          }
       #else
               for (int j=0; j<NY+2; j++){
              y[j] = ( (float) j - 0.5) * dy;
              }
       #endif

  /*** Set up Gaussian initial conditions ***/
  /* LOOP 3  >> parallelized*/
       #ifdef _OPENMP
        #pragma omp parallel for default (none) shared(u) collapse(2)
          for (int i=0; i<NX+2; i++){
            for (int j=0; j<NY+2; j++){
              u[i][j] = 0;
            }
          }
      #else
         for (int i=0; i<NX+2; i++){
            for (int j=0; j<NY+2; j++){
              u[i][j] = 0;
            }
          }
      #endif

  /*** Write array of initial u values out to file ***/
  FILE *initialfile;
  initialfile = fopen("initial.dat", "w");
  /* LOOP 4  >> Bellow nested loop can not be parallelized because we want to print out the results of our calculation
                to a file in order so there is an "output dependency" */

  for (int i=0; i<NX+2; i++){
    for (int j=0; j<NY+2; j++){
      fprintf(initialfile, "%g %g %g\n", x[i], y[j], u[i][j]);
    }
  }
  fclose(initialfile);

    /*** Update solution by looping over time steps ***/
    /* LOOP 5 >> This loop has Output Dependency so we should set private(dudt,VxVAR) lastprivate(u) to be parallelized. But I got an error as:
    (Segmentation fault (core dumped)) because the system need more memory to solve it. So I added required syntax as bellow but comment it out to run correctly
     so it will be run sequential.*/
    //#pragma omp parallel for default(none) shared(dx,dy,y,dt) private(dudt,VxVAR) lastprivate(u)
    for (int m=0; m<nsteps; m++){

    /*** Apply boundary conditions at u[0][:] and u[NX+1][:] ***/
    /* LOOP 6 >> Parallelized + Modification of Boundary condition at x = 0 (the left boundary) */
        #ifdef _OPENMP
           #pragma omp parallel for default (none) shared(u,m,y,dt)
                for (int j=0; j<NY+2; j++){
                  u[0][j]    = exp((-(pow((y[j]-y0),2))/(2*sigmay2))-(pow((m*dt-t0),2)/(2*sigmat2)));
                  u[NX+1][j] = bval_right;
                }
        #else
                for (int j=0; j<NY+2; j++){
                  u[0][j]    = exp((-(pow((y[j]-y0),2))/(2*sigmay2))-(pow((m*dt-t0),2)/(2*sigmat2)));
                  u[NX+1][j] = bval_right;
                }
        #endif

    /*** Apply boundary conditions at u[:][0] and u[:][NY+1] ***/
    /* LOOP 7 >> Parallelized */
        #ifdef _OPENMP
        #pragma omp parallel for default (none) shared(u)
            for (int i=0; i<NX+2; i++){
              u[i][0]    = bval_lower;
              u[i][NY+1] = bval_upper;
            }
        #else
                for (int i=0; i<NX+2; i++){
                  u[i][0]    = bval_lower;
                  u[i][NY+1] = bval_upper;
                }
        #endif

    /*** Calculate rate of change of u using leftward difference ***/
    /* Loop over points in the domain but not boundary values */
    /* LOOP 8 >> Parallelized */
        #ifdef _OPENMP
          #pragma omp parallel for default (none) shared(dx,dy,dudt,u,y) private(VxVAR) collapse(2)
            for (int i=1; i<NX+1; i++){
              for (int j=1; j<NY+1; j++){
                if(y[j]>z0){
                    VxVAR = (u_star/k) * log(y[j]/z0);
                }
                else if(y[j]<=z0){
                    VxVAR = 0;
                }
              dudt[i][j] = - VxVAR * (u[i][j] - u[i-1][j]) / dx - vely * (u[i][j] - u[i][j-1]) / dy;
              }
            }
        #else
            for (int i=1; i<NX+1; i++){
              for (int j=1; j<NY+1; j++){
                if(y[j]>z0){
                    VxVAR = (u_star/k) * log(y[j]/z0);
                }
                else if(y[j]<=z0){
                    VxVAR = 0;
                }
              dudt[i][j] = - VxVAR * (u[i][j] - u[i-1][j]) / dx - vely * (u[i][j] - u[i][j-1]) / dy;
              }
            }
        #endif

    /*** Update u from t to t+dt ***/
    /* Loop over points in the domain but not boundary values */
    /* LOOP 9 >> Parallelized */
        #ifdef _OPENMP
          #pragma omp parallel for default (none) shared(u,dudt,dt) collapse(2)
            for	(int i=1; i<NX+1; i++){
              for (int j=1; j<NY+1; j++){
            u[i][j] = u[i][j] + dudt[i][j] * dt;
              }
            }
        #else
            for	(int i=1; i<NX+1; i++){
              for (int j=1; j<NY+1; j++){
            u[i][j] = u[i][j] + dudt[i][j] * dt;
              }
            }
        #endif

  } // time loop

  /*** Write array of final u values out to file ***/
  FILE *finalfile;
  finalfile = fopen("final.dat", "w");
  /* LOOP 10  >> Bellow nested loop can not be parallelized because we want to print out the results of our calculation
                to a file in order so there is an "output dependency" */
    for (int i=0; i<NX+2; i++){
        for (int j=0; j<NY+2; j++){
        fprintf(finalfile, "%g %g %g\n", x[i], y[j], u[i][j]);
        }
    }
    fclose(finalfile);

/**~~~~~~~~~~~ Task 2.4 :calculation the vertically averaged distribution of u(x, y) ~~~~~~~~~~~***/
/* Below nested loop calculates the vertically averaged distribution of u(x,y) over points in the domain but not boundary
   values. */
    for(int i = 1; i<= NX ; i++){
        for(int j = 1; j<= NY ; j++){
            temp_sum =temp_sum+u[i][j];
        }
        vert_avg[i]=temp_sum/NY;
        temp_sum=0;
    }
  FILE *AVGfile;
  AVGfile = fopen("VerticalAVG.dat", "w");

  /* creating the "VerticalAVG.dat" to be used in line graph plot. */
  for (int i=1; i<NX+1; i++){
      fprintf(AVGfile, "%g %g\n", x[i], vert_avg[i]);
    }
  fclose(AVGfile);

  return 0;
}

/* End of file ******************************************************/
