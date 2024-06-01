# include <math.h>
# include <stdio.h>
# include <stdlib.h>
# include <time.h>
# include <sys/time.h>
# include <omp.h> // Agregamos la librerÃ­a OpenMP

int main ( int argc, char *argv[] );
void compute ( int np, int nd, double pos[], double vel[], 
               double mass, double f[], double *pot, double *kin );
double dist ( int nd, double r1[], double r2[], double dr[] );
void initialize ( int np, int nd, double pos[], double vel[], double acc[] );
void r8mat_uniform_ab ( int m, int n, double a, double b, int *seed, double r[] );
void update ( int np, int nd, double pos[], double vel[], double f[], 
             double acc[], double mass, double dt );

int main ( int argc, char *argv[] )
{
  double *acc;
  double ctime;
  double dt;
  double e0;
  double *force;
  double kinetic;
  double mass = 1.0;
  int nd;
  int np;
  double *pos;
  double potential;
  int step;
  int step_num;
  int step_print;
  int step_print_index;
  int step_print_num;
  double *vel;

  struct timeval tv_start, tv_end;
  double run_time;

  printf ( "\n" );
  printf ( "MD\n" );
  printf ( "  C version\n" );
  printf ( "  A molecular dynamics program.\n" );

  if ( 1 < argc )
    nd = atoi ( argv[1] );
  else
  {
    printf ( "\n" );
    printf ( "  Enter ND, the spatial dimension (2 or 3).\n" );
    scanf ( "%d", &nd );
  }

  if ( 2 < argc )
    np = atoi ( argv[2] );
  else
  {
    printf ( "\n" );
    printf ( "  Enter NP, the number of particles (500, for instance).\n" );
    scanf ( "%d", &np );
  }

  if ( 3 < argc )
    step_num = atoi ( argv[3] );
  else
  {
    printf ( "\n" );
    printf ( "  Enter ND, the number of time steps (500 or 1000, for instance).\n" );
    scanf ( "%d", &step_num );
  }

  if ( 4 < argc )
    dt = atof ( argv[4] );
  else
  {
    printf ( "\n" );
    printf ( "  Enter DT, the size of the time step (0.1, for instance).\n" );
    scanf ( "%lf", &dt );
  }

  printf ( "\n" );
  printf ( "  ND, the spatial dimension, is %d\n", nd );
  printf ( "  NP, the number of particles in the simulation, is %d\n", np );
  printf ( "  STEP_NUM, the number of time steps, is %d\n", step_num );
  printf ( "  DT, the size of each time step, is %f\n", dt );

  acc = ( double * ) malloc ( nd * np * sizeof ( double ) );
  force = ( double * ) malloc ( nd * np * sizeof ( double ) );
  pos = ( double * ) malloc ( nd * np * sizeof ( double ) );
  vel = ( double * ) malloc ( nd * np * sizeof ( double ) );

  printf ( "\n" );
  printf ( "  At each step, we report the potential and kinetic energies.\n" );
  printf ( "  The sum of these energies should be a constant.\n" );
  printf ( "  As an accuracy check, we also print the relative error\n" );
  printf ( "  in the total energy.\n" );
  printf ( "\n" );
  printf ( "      Step      Potential       Kinetic        (P+K-E0)/E0\n" );
  printf ( "                Energy P        Energy K       Relative Energy Error\n" );
  printf ( "\n" );

  step_print = 0;
  step_print_index = 0;
  step_print_num = 10;

  gettimeofday(&tv_start, NULL);


            for (step = 0; step <= step_num; step++)
            {
                if (step == 0)
                    initialize(np, nd, pos, vel, acc);
                else
                    update(np, nd, pos, vel, force, acc, mass, dt);

                compute(np, nd, pos, vel, mass, force, &potential, &kinetic);

                if (step == 0)
                    e0 = potential + kinetic;

                if (step == step_print)
                {
                    printf("  %8d  %14f  %14f  %14e\n", step, potential, kinetic,
                           (potential + kinetic - e0) / e0);
                    step_print_index = step_print_index + 1;
                    step_print = (step_print_index * step_num) / step_print_num;
                }
            }

  gettimeofday(&tv_end, NULL);
  run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 +
        (tv_end.tv_usec - tv_start.tv_usec);
  run_time = run_time/1000000;

  printf("\n Tiempo version OpenMP = %lg s\n", run_time);

  free ( acc );
  free ( force );
  free ( pos );
  free ( vel );

  printf ( "\n" );
  printf ( "MD\n" );
  printf ( "  Normal end of execution.\n" );
  printf ( "\n" );

  return 0;
}

void compute(int np, int nd, double pos[], double vel[], double mass, 
             double f[], double *pot, double *kin) {
    double d;
    double d2;
    int i, j, k;
    double ke = 0.0;
    double pe = 0.0;
    double PI2 = 1.570796327;
    double rij[3];

    pe = 0.0;
    ke = 0.0;

    #pragma omp target map(tofrom:pe,ke,f[:np*nd]) map(to:pos[:np*nd],vel[:np*nd])
    #pragma omp teams distribute parallel for private(i, j, d, d2, rij) reduction(+:pe,ke)
    for (k = 0; k < np; k++) {
        // Inicializar fuerzas en 0 para cada partícula k
        for (i = 0; i < nd; i++)
            f[i + k * nd] = 0.0;

        for (j = 0; j < np; j++) {
            if (k != j) {
                d = dist(nd, pos + k * nd, pos + j * nd, rij);

                if (d < PI2)
                    d2 = d;
                else
                    d2 = PI2;

                pe += 0.5 * pow(sin(d2), 2);

                for (i = 0; i < nd; i++)
                    f[i + k * nd] -= rij[i] * sin(2.0 * d2) / d;
            }
        }

        // Calcular energía cinética para cada partícula k
        double local_ke = 0.0;
        for (i = 0; i < nd; i++)
            local_ke += vel[i + k * nd] * vel[i + k * nd];

        ke += local_ke;
    }

    ke *= 0.5 * mass;

    *pot = pe;
    *kin = ke;
}




double dist ( int nd, double r1[], double r2[], double dr[] )
{
  double d;
  int i;

  d = 0.0;
  for ( i = 0; i < nd; i++ )
  {
    dr[i] = r1[i] - r2[i];
    d = d + dr[i] * dr[i];
  }
  d = sqrt ( d );

  return d;
}

void initialize ( int np, int nd, double pos[], double vel[], double acc[] )
{
  int i;
  int j;
  int seed;

  seed = 123456789;
  r8mat_uniform_ab ( nd, np, 0.0, 10.0, &seed, pos );

  for ( j = 0; j < np; j++ )
  {
    for ( i = 0; i < nd; i++ )
    {
      vel[i+j*nd] = 0.0;
      acc[i+j*nd] = 0.0;
    }
  }   


  return;
}

void r8mat_uniform_ab ( int m, int n, double a, double b, int *seed, double r[] )
{
  int i;
  const int i4_huge = 2147483647;
  int j;
  int k;

  if ( *seed == 0 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "R8MAT_UNIFORM_AB - Fatal error!\n" );
    fprintf ( stderr, "  Input value of SEED = 0.\n" );
    exit ( 1 );
  }

  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      k = *seed / 127773;

      *seed = 16807 * ( *seed - k * 127773 ) - k * 2836;

      if ( *seed < 0 )
        *seed = *seed + i4_huge;
      r[i+j*m] = a + ( b - a ) * ( double ) ( *seed ) * 4.656612875E-10;
    }
  }

  return;
}

void update ( int np, int nd, double pos[], double vel[], double f[], 
  double acc[], double mass, double dt )
{
  int i;
  int j;
  double rmass;

  rmass = 1.0 / mass;

  for ( j = 0; j < np; j++ )
  {
    for ( i = 0; i < nd; i++ )
    {
      pos[i+j*nd] = pos[i+j*nd] + vel[i+j*nd] * dt + 0.5 * acc[i+j*nd] * dt * dt;
      vel[i+j*nd] = vel[i+j*nd] + 0.5 * dt * ( f[i+j*nd] * rmass + acc[i+j*nd] );
      acc[i+j*nd] = f[i+j*nd] * rmass;
    }
  }

  return;
}
