/*
 * Jonathan Senning <jonathan.senning@gordon.edu>
 * Department of Mathematics and Computer Science
 * Gordon College, 255 Grapevine Road, Wenham MA 01984-1899
 * August 2012, April 2013
 *
 * $Smake: mpicc -DGATHER_DATA -DWRITE_FILE -Wall -O2 -o %F %f -lhdf5 -lz
 *
 * This program demonstrates how to set up and use a Cartesian grid
 * when data along the boundary between neighboring regions must be
 * exchanged.  It illustrates MPI_Gather(), MPI_Gatherv() and how to
 * write a data file in parallel using HDF5.
 *
 * Suppose a rectangular 10x10 two-dimensional domain is to be
 * partitioned into four subdomains.  Often computation on the domain
 * requires the stencil
 *
 *                 *
 *                 |
 *             *---X---*
 *                 |
 *                 *
 *
 * when updating grid points that are on the boundary of a subdomain
 * that are adjacent to another subdomain.  This can be facilitated by
 * "adding" additional rows and columns to subdomains that will hold
 * copies of the interior boundary grid points from adjacent subdomains.
 *
 * Here is the unpartitioned grid, showing the boundary grid points
 * ("*") whose data is used but unchanged, and the grid points on the
 * domain interior ("O") that are updated by computation.
 *
 *      0   1   2   3   4   5   6   7   8   9
 * 
 *   0  *---*---*---*---*---*---*---*---*---*
 *      |   |   |   |   |   |   |   |   |   |
 *   1  *---O---O---O---O---O---O---O---O---*
 *      |   |   |   |   |   |   |   |   |   |
 *   2  *---O---O---O---O---O---O---O---O---*
 *      |   |   |   |   |   |   |   |   |   |
 *   3  *---O---O---O---O---O---O---O---O---*    "*" is boundry node
 *      |   |   |   |   |   |   |   |   |   |
 *   4  *---O---O---O---O---O---O---O---O---*    "O" is interior node
 *      |   |   |   |   |   |   |   |   |   |
 *   5  *---O---O---O---O---O---O---O---O---*
 *      |   |   |   |   |   |   |   |   |   |
 *   6  *---O---O---O---O---O---O---O---O---*
 *      |   |   |   |   |   |   |   |   |   |
 *   7  *---O---O---O---O---O---O---O---O---*
 *      |   |   |   |   |   |   |   |   |   |
 *   8  *---O---O---O---O---O---O---O---O---*
 *      |   |   |   |   |   |   |   |   |   |
 *   9  *---*---*---*---*---*---*---*---*---*
 *
 * Here are the four partitioned grid parts.  In each grid subdomain,
 * "O,A,B,C,D" represent grid points that are computed.  The points
 * labeled "A,B,C,D" are the grid points adjacent to an interior
 * boundary so they must be copied to added positions in adjacent
 * subdomains, indicated by "a,b,c,d".
 *
 *      0   1   2   3   4  5   4  5   6   7   8   9
 * 
 *   0  *---*---*---*---*--*   *--*---*---*---*---* 
 *      |   |   |   |   |         |   |   |   |   |
 *   1  *---O---O---O---A--b   a--B---O---O---O---*
 *      |   |   |   |   |         |   |   |   |   |
 *   2  *---O---O---O---A--b   a--B---O---O---O---*
 *      |   |   |   |   |         |   |   |   |   |
 *   3  *---O---O---O---A--b   a--B---O---O---O---*   "*" is boundry node
 *      |   |   |   |   |         |   |   |   |   |
 *   4  *---A---A---A---A--b   a--B---B---B---B---*   "O" is interior node
 *      |   |   |   |   |         |   |   |   |   |
 *   5  *   d   d   d   d         c   c   c   c   *   "A,B,C,D" are computed
 *                                                    interior boundary nodes
 *   4  *   a   a   a   a         b   b   b   b   *
 *      |   |   |   |   |         |   |   |   |   |   "a,b,c,d" are copies of
 *   5  *---D---D---D---D--c   d--C---C---C---C---*   interior boundary nodes
 *      |   |   |   |   |         |   |   |   |   |
 *   6  *---O---O---O---D--c   d--C---O---O---O---*
 *      |   |   |   |   |         |   |   |   |   |
 *   7  *---O---O---O---D--c   d -C---O---O---O---*
 *      |   |   |   |   |         |   |   |   |   |
 *   8  *---O---O---O---D--c   d--C---O---O---O---*
 *      |   |   |   |   |         |   |   |   |   |
 *   9  *---*---*---*---*--*   *--*---*---*---*---*
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>
#include <assert.h>

#if defined(WRITE_FILE)
# include <hdf5.h>
#endif

#if defined(SYNC)
# define SYNCHRONOUS
#endif

typedef struct Region {
    int x0;
    int x1;
    int y0;
    int y1;
    int nx;
    int ny;
} Region;

/*----------------------------------------------------------------------------
 * Given an array with n elements that is to be partitioned into m
 * subintervals of approximately equal size, this function returns the
 * first and last index (inclusive) of the ith subinterval.  For
 * example, the code
 *     decompose1d( 100, 3, 0, &s, &e );
 *     printf( "%2d %2d\n", s, e );
 *     decompose1d( 100, 3, 1, &s, &e );
 *     printf( "%2d %2d\n", s, e );
 *     decompose1d( 100, 3, 2, &s, &e );
 *     printf( "%2d %2d\n", s, e );
 * will display
 *      0 33
 *     34 66
 *     67 99
 *
 * Input:
 *    int n        - length of array (array indexed [0]..[n-1])
 *    int m        - number of subintervals
 *    int i        - subinterval number
 *
 * Output:
 *    int* s       - location to store subinterval starting index
 *    int* e       - location to store subinterval ending index
 *
 * This function is based on the FORTRAN subroutine MPE_DECOMP1D in the
 * file UsingMPI/intermediate/decomp.f supplied with the book Using MPI
 * by Gropp et al.  It has been adapted to use 0-based indexing.
 */
void decompose1d( int n, int m, int i, int* s, int* e )
{
    const int length  = n / m;
    const int deficit = n % m;
    *s =  i * length + ( i < deficit ? i : deficit );
    *e = *s + length - ( i < deficit ? 0 : 1 );
    if ( ( *e >= n ) || ( i == m - 1 ) ) *e = n - 1;
}

/*----------------------------------------------------------------------------
 * Exchange interior boundary data between 2D grid blocks.
 *
 * Input:
 *   double* v       - two-dimensional array holding single block of grid data
 *   int nx, ny      - dimensions of grid block
 *   int up, down    - ranks of processes handling blocks above and below ours
 *   int left, right - ranks of processes handling blocks to each side of ours
 *   MPI_Datatype xSlice - type representing horizontal row of data
 *   MPI_Datatype ySlize - type representing vertical column of data
 *   MPI_Comm comm       - communicator
 *   MPI_Request sndreq  - send request handle (used for async. send)
 *   MPI_Request rcvreq  - receive request handle (used for async. recv)
 *
 * Output:
 *   double* v       - two-dimensional array holding single block of grid data
 */
void exchangeSlices( double* u, int nx, int ny, 
                     int up, int down, int left, int right, 
                     MPI_Datatype xSlice, MPI_Datatype ySlice, MPI_Comm comm,
                     MPI_Request* sndreq, MPI_Request* rcvreq )
{
    enum { T1, T2, T3, T4 };

#if defined(SYNCHRONOUS)    
    /* Exchange x-slices with my top and bottom neighbors */
    MPI_Sendrecv( &u[(0) * ny + (ny-2)], 1, xSlice, up,   T1,
                  &u[(0) * ny + (0)],    1, xSlice, down, T1,
                  comm, MPI_STATUS_IGNORE );
    MPI_Sendrecv( &u[(0) * ny + (1)],    1, xSlice, down, T2,
                  &u[(0) * ny + (ny-1)], 1, xSlice, up,   T2,
                  comm, MPI_STATUS_IGNORE );

    /* Exchange y-slices with my left and right neighbors */
    MPI_Sendrecv( &u[(nx-2) * ny + (0)], 1, ySlice, right, T3,
                  &u[(0) * ny + (0)],    1, ySlice, left,  T3,
                  comm, MPI_STATUS_IGNORE );
    MPI_Sendrecv( &u[(1) * ny + (0)],    1, ySlice, left,  T4,
                  &u[(nx-1) * ny + (0)], 1, ySlice, right, T4,
                  comm, MPI_STATUS_IGNORE );
#else
    /* Exchange x-slices with my top and bottom neighbors */
    MPI_Isend( &u[(0) * ny + (ny-2)], 1, xSlice, up,   T1, comm, &sndreq[0] );
    MPI_Irecv( &u[(0) * ny + (0)],    1, xSlice, down, T1, comm, &rcvreq[0] );
    MPI_Isend( &u[(0) * ny + (1)],    1, xSlice, down, T2, comm, &sndreq[1] );
    MPI_Irecv( &u[(0) * ny + (ny-1)], 1, xSlice, up,   T2, comm, &rcvreq[1] );

    /* Exchange y-slices with my left and right neighbors */
    MPI_Isend( &u[(nx-2) * ny + (0)], 1, ySlice, right, T3, comm, &sndreq[2] );
    MPI_Irecv( &u[(0) * ny + (0)],    1, ySlice, left,  T3, comm, &rcvreq[2] );
    MPI_Isend( &u[(1) * ny + (0)],    1, ySlice, left,  T4, comm, &sndreq[3] );
    MPI_Irecv( &u[(nx-1) * ny + (0)], 1, ySlice, right, T4, comm, &rcvreq[3] );
#endif
}

/*----------------------------------------------------------------------------
 * Print out 2D grid data
 *
 * Input:
 *   double* v       - two-dimensional array holding grid data
 *   int nx, ny      - dimensions of grid
 * Output:
 *   None, other than output to stdout.
 */
void showGrid( double* v, int nx, int ny )
{
    int i, j;

    printf( "------------------------------------------------------------\n" );
    for ( j = ny - 1; j >= 0; j-- )
    {
        for ( i = 0; i < nx; i++ )
        {
            printf( " %6.4f", v[i * ny + j] );
        }
        printf( "\n" );
    }
    printf( "------------------------------------------------------------\n" );
}

#if defined(GATHER_DATA)
/*----------------------------------------------------------------------------
 * Gather and display data
 *
 * Input
 *   MPI_Comm comm     - MPI communicator
 *   int rank          - rank of current process
 *   int num_proc      - total number of processes
 *   double* u         - local portion of array to write to file
 *   Region* orig_grid - local grid specification
 *   Region* halo_grid - local grid with ghost boundaries specification
 *   int NX, NY        - dimensions of overall array
 * Output
 *   None, other than displayed data
 */
void gatherData( MPI_Comm comm, int rank, int num_proc, double* u,
                 Region* orig_grid, Region* halo_grid, int NX, int NY )
{
    Region* domainList = NULL; /* list of specs for all grid blocks */
    int* domainSize = NULL;  /* size for each grid block */
    int* domainDisp = NULL;  /* displacement to start of each grid block */
    double* sendbuf = NULL;  /* holds grid data to be sent to master */
    double* recvbuf = NULL;  /* buffer to collect grid data from all blocks */
    int n;                   /* buffer index */
    int i, j, k;

    /*
     * Now we want to collect entire entire grid data into rank 0
     * process.  To do this we:
     * 1. Gather grid block dimension information from each process
     * 2. Compute block grid sizes and displacements
     * 3. Fill local send buffer; we only send true boundary nodes and
     *      nodes on the interior of the subdomain
     * 4. Gather all grid data into single buffer on rank 0 process
     */
    domainList = (Region*) malloc( num_proc * sizeof( Region ) );
    domainSize = (int*) malloc( num_proc * sizeof( int ) );
    domainDisp = (int*) malloc( num_proc * sizeof( int ) );

    /*
     * 1. Gather grid block dimension information from each process
     */
    k = sizeof( Region ) / sizeof( int );
    MPI_Gather( orig_grid, k, MPI_INT, domainList, k, MPI_INT, 0, comm );

    /*
     * 2. Compute block grid sizes and displacements
     */
    domainSize[0] = domainList[0].nx * domainList[0].ny;
    domainDisp[0] = 0;
    for ( k = 1; k < num_proc; k++ )
    {
        domainSize[k] = domainList[k].nx * domainList[k].ny;
        domainDisp[k] = domainDisp[k-1] + domainSize[k-1];
    }

    /*
     * 3. Fill local send buffer; we only send true boundary nodes and
     *      nodes on the interior of the subdomain
     */
    sendbuf = (double*) malloc( orig_grid->nx * orig_grid->ny * sizeof(double));

    n = 0;
    for ( i = orig_grid->x0 - halo_grid->x0; 
          i <= orig_grid->x1 - halo_grid->x0; i++ )
    {
        for ( j = orig_grid->y0 - halo_grid->y0; 
              j <= orig_grid->y1 - halo_grid->y0; j++ )
        {
            sendbuf[n++] = u[i * halo_grid->ny + j];
        }
    }

    /*
     * 4. Gather all grid data into single buffer on rank 0 process.
     * Notice that this buffer is only allocated on the rank 0 process.
     * Note also that we used use MPI_Gatherv() since the amount of
     * data being gathered from each process can vary.
     */
    if ( rank == 0 ) recvbuf = (double*) malloc( NX * NY * sizeof( double ) );
    MPI_Gatherv( sendbuf, n, MPI_DOUBLE,
                 recvbuf, domainSize, domainDisp, MPI_DOUBLE, 0, comm );

    /*
     * Rank 0 process displays data it's been sent.  Although recvbuf
     * contains all the data, it may not be in the order we expect it.
     * Using information in domainList we can figure out the layout.
     * Here we just create another buffer to hold all the data and copy
     * from recvbuf in the order we want.  This is not very efficient, and
     * in a real program some effort should be made to avoid doing this.
     */
    if ( rank == 0 )
    {
        /*
         * allocate yet another buffer to hold reorganized data
         */
        double* z = (double*) malloc( NX * NY * sizeof( double ) );

        /*
         * copy data into new buffer
         */
        n = 0;
        for ( k = 0; k < num_proc; k++ )
        {
            for ( i = domainList[k].x0; i <= domainList[k].x1; i++ )
            {
                for ( j = domainList[k].y0; j <= domainList[k].y1; j++ )
                {
                    z[i * NY + j] = recvbuf[n++];
                }
            }
        }

        /*
         * display data
         */
        printf( "Values have the form R.XXYY where:\n" );
        printf( "\tR  is the rank of of the process that created the data\n" );
        printf( "\tXX is the x coordinate in the grid (0 is at left)\n" );
        printf( "\tYY is the y coordinate in the grid (0 is at bottom)\n" );
        showGrid( z, NX, NY );
        free( z );
    }
    free( sendbuf ); 
    free( recvbuf );
    free( domainList );
    free( domainSize );
    free( domainDisp );
}
#endif

#if defined(WRITE_FILE)
/*----------------------------------------------------------------------------
 * Create HDF5 file and store grid data in it.
 *
 * Input
 *   char* fname       - name of file to create
 *   char* dname       - name of dataset in file
 *   MPI_Comm comm     - MPI communicator used by processes sharing in write
 *   double* u         - local portion of array to write to file
 *   Region* orig_grid - local grid specification
 *   Region* halo_grid - local grid with ghost boundaries specification
 *   int NX, NY        - dimensions of overall array
 * Output
 *   None, other than created file
 */
void writeFile( char* fname, char* dname, MPI_Comm comm, double* u,
                Region* orig_grid, Region* halo_grid, int NX, int NY )
{
    hid_t plist_id;       /* property list */
    hid_t file_id;        /* output file */
    hid_t dataspace_id;   /* dataspace of output file */
    hid_t dataset_id;     /* dataset in output file */
    hid_t memspace_id;    /* dataspace of local memory */
    hsize_t dimsf[2];     /* dimensions of dataspace in file */
    hsize_t dimsm[2];     /* dimensions of dataspace in memory */
    hsize_t count[2];     /* number of blocks */
    hsize_t offset[2];    /* start values of data block locations */
    herr_t status;
    const MPI_Info info = MPI_INFO_NULL;

    /*
     * Create property list and store MPI IO communicator information in it.
     */
    plist_id = H5Pcreate( H5P_FILE_ACCESS );
    status = H5Pset_fapl_mpio( plist_id, comm, info );
    assert( status >= 0 );

    /*
     * Create the output file (truncate if preexisting) using property list.
     */
    file_id = H5Fcreate( fname, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id );
    assert( file_id >= 0 );
    H5Pclose( plist_id );

    /*
     * Create output file dataspace and dataset that will hold data from all
     * processes.  Use dimensions for entire grid.
     */
    dimsf[0] = NX;
    dimsf[1] = NY;
    dataspace_id = H5Screate_simple( 2, dimsf, NULL );
    assert( dataspace_id >= 0 );
    dataset_id = H5Dcreate( file_id, dname, H5T_IEEE_F64LE, dataspace_id,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
    assert( dataset_id >= 0 );

    /*
     * Create local dataspace corresponding to process's portion of the
     * global dataspace.  Use the actual dimensions for the portion of
     * the data this process is responsible for, including ghost boundaries.
     */
    dimsm[0] = halo_grid->nx;
    dimsm[1] = halo_grid->ny;
    memspace_id = H5Screate_simple( 2, dimsm, NULL );
    assert( memspace_id >= 0 );

    /*
     * Define the hyperslab in the file's dataspace
     */
    count[0] = orig_grid->nx;
    count[1] = orig_grid->ny;
    offset[0] = orig_grid->x0;
    offset[1] = orig_grid->y0;
    status = H5Sselect_hyperslab( dataspace_id, H5S_SELECT_SET,
                                  offset, NULL, count, NULL );
    assert( status >= 0 );

    /*
     * Define the hyperslab in the memory dataspace.  Offsets are nonzero
     * if there is a ghost boundary to ignore.
     */
    offset[0] = orig_grid->x0 - halo_grid->x0;
    offset[1] = orig_grid->y0 - halo_grid->y0;
    status = H5Sselect_hyperslab( memspace_id, H5S_SELECT_SET,
                                  offset, NULL, count, NULL );
    assert( status >= 0 );
 
    /*
     * Set transfer mode to be collective (rather than independent)
     */
    plist_id = H5Pcreate( H5P_DATASET_XFER );
    assert( plist_id >= 0 );
    status = H5Pset_dxpl_mpio( plist_id, H5FD_MPIO_COLLECTIVE );
    assert( status >= 0 );    

    /*
     * Finally - we can write the data!
     */
    status = H5Dwrite( dataset_id, H5T_NATIVE_DOUBLE, memspace_id,
                       dataspace_id, plist_id, u );
    assert( status >= 0 );

    /*
     * All done -- release all remaining open dataspaces, datasets, etc.
     */
    H5Pclose( plist_id );
    H5Sclose( memspace_id );
    H5Dclose( dataset_id );
    H5Sclose( dataspace_id );
    H5Fclose( file_id );
}
#endif

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

int main( int argc, char *argv[] )
{
    int NX = 10;              /* number of grid points in x direction */
    int NY = 10;              /* number of grid points in y direction */
    int num_proc;             /* number of participating processes */
    int rank;                 /* process rank within communicator */
    MPI_Comm comm2d;          /* Cartesian communicator */
    int dims[2] = { 0, 0 };   /* allow MPI to choose grid block dimensions */
    int periodic[2] = { 0, 0 };  /* domain is non-periodic */
    int reorder = 1;          /* allow processes to be re-ranked */
    int coords[2];            /* coordinates of our block in grid */
    int up, down;             /* ranks of processes above and below ours */
    int left, right;          /* ranks of processes to each side of ours */
    Region orig_grid;         /* original grid block specification */
    Region halo_grid;         /* grid block with halo (ghost boundaries) */
    double* u = NULL;         /* array to hold grid block data */
    MPI_Datatype xSlice;      /* datatype for horizontal slice (single row) */
    MPI_Datatype ySlice;      /* datatype for vertical slice (single column) */
    int i, j;                 /* counters */
    MPI_Request sndreq[4];
    MPI_Request rcvreq[4];

    /*
     * Process command line: two arguments (plus program name) are required,
     * third and fourth are optional but must both appear if either does.
     */
    if ( argc == 2 || argc == 4 )
    {
        printf( "Usage: %s [NX NY [DIMX DIMY]]\n\n", argv[0] );
        printf( "- NX and NY are the grid dimensions.\n" );
        printf( "- DIMX and DIMY are the process grid dimensions.\n" );
        exit( EXIT_FAILURE );
    }

    /*
     * Override default grid dimensions if requested
     */
    if ( argc > 2 )
    {
        NX = atoi( argv[1] );
        NY = atoi( argv[2] );
    }
    if ( NX <= 0 || NY <= 0 )
    {
        if ( rank == 0 )
        {
            fprintf( stderr, "Error: both NX and NY must be positive.\n" );
        }
        exit( EXIT_FAILURE );
    }

    /*
     * Override default process grid dimensions if requested
     */
    if ( argc > 4 )
    {
        dims[0] = atoi( argv[3] );
        dims[1] = atoi( argv[4] );
    }

    /*
     * Initialize MPI and make sure that if the user specified block
     * grid dimensions they are consistent with the number of processes.
     */
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &num_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    if ( dims[0] * dims[1] > 0 && dims[0] * dims[1] != num_proc )
    {
        if ( rank == 0 )
        {
            fprintf( stderr, "Product of grid block dimensions must match " );
            fprintf( stderr, "number of processes\n" );
        }
        MPI_Finalize();
        exit( EXIT_FAILURE );
    }

    /*
     * Set up Cartesian grid of processors.  A new communicator is
     * created we get our rank within it.
     */
    MPI_Dims_create( num_proc, 2, dims );
    MPI_Cart_create( MPI_COMM_WORLD, 2, dims, periodic, reorder, &comm2d );
    MPI_Cart_get( comm2d, 2, dims, periodic, coords );
    MPI_Comm_rank( comm2d, &rank );

    /*
     * Figure out who my neighbors are.  left, right, down, and up will
     * be set to the rank of the process responsible for the corresponding
     * block relative to the position of the block we are responsible for.
     * If there is no neighbor in a particular direction the returned rank
     * will be MPI_PROC_NULL which will be ignored by subsequent
     * MPI_sendrecv() calls.
     */
    MPI_Cart_shift( comm2d, 0, 1, &left, &right );
    MPI_Cart_shift( comm2d, 1, 1, &down, &up );

    /*
     * Figure out the size of my portion of the grid.  Note that we adjust
     * the starting and ending grid indices along boundaries with neighboring
     * subdomains; this provides room to hold the internal "boundary" data
     * that will be received from the neighbor.
     */
    decompose1d( NX, dims[0], coords[0], &orig_grid.x0, &orig_grid.x1 );
    decompose1d( NY, dims[1], coords[1], &orig_grid.y0, &orig_grid.y1 );
    orig_grid.nx = orig_grid.x1 - orig_grid.x0 + 1;
    orig_grid.ny = orig_grid.y1 - orig_grid.y0 + 1;

    /*
     * Compute adjusted domain paramters to account for inter-domain
     * boundary data.  If we have a neighbor in a given direction
     * (rank of neighbor is non- negative) then we need to adjust the
     * starting or ending index.
     */
    halo_grid.x0 = orig_grid.x0 - (  left >= 0 ? 1 : 0 );
    halo_grid.x1 = orig_grid.x1 + ( right >= 0 ? 1 : 0 );
    halo_grid.y0 = orig_grid.y0 - (  down >= 0 ? 1 : 0 );
    halo_grid.y1 = orig_grid.y1 + (    up >= 0 ? 1 : 0 );
    halo_grid.nx = halo_grid.x1 - halo_grid.x0 + 1;
    halo_grid.ny = halo_grid.y1 - halo_grid.y0 + 1;

    /*
     * Create my portion of the grid.  We will assume grid is stored in
     * usual C fashion; second index address contiguous values.
     */
    u = (double*) malloc( halo_grid.nx * halo_grid.ny * sizeof( double ) );

    /*
     * Since this is a demonstration program, here we initialize the
     * grid with values that indicate their original position in the
     * grid.  Assuming NX and NY are 100 or smaller then these values
     * have the form R.XXYY where:
     *     R  is the rank of of the process that created the data
     *     XX is the x coordinate in the grid (0 is at left)
     *     YY is the y coordinate in the grid (0 is at bottom)
     */
    for ( j = 0; j < halo_grid.ny; j++ )
    {
        for ( i = 0; i < halo_grid.nx; i++ )
        {
            u[i * halo_grid.ny + j] = rank
                + 0.01   * ( i + halo_grid.x0 ) 
                + 0.0001 * ( j + halo_grid.y0 );
        }
    }

    /*
     * Create datatypes for exchanging x and y slices
     */
    MPI_Type_vector( halo_grid.nx, 1, halo_grid.ny, MPI_DOUBLE, &xSlice );
    MPI_Type_commit( &xSlice );

    MPI_Type_vector( halo_grid.ny, 1, 1, MPI_DOUBLE, &ySlice );
    MPI_Type_commit( &ySlice );

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /*
     * Now we're done setting things up.  In a "real" program, now we'd
     * begin doing whatever we need to do with the data.  In this case, we
     * merely exchange the slices to copy interior boundary data to adjacent
     * processes.
     */
    exchangeSlices( u, halo_grid.nx, halo_grid.ny, up, down, left, right,
                    xSlice, ySlice, comm2d, sndreq, rcvreq );

    /*
     * Here we would do work that can be done prior to the exchange of
     * the slices being completed.  Typically this would mean updates
     * to all of the interior of the domain except the cells adjacent
     * to the ghost boundaries
     */

#if !defined(SYNCHRONOUS)
    /*
     * Wait for ghost boundary data to arrive
     */
    MPI_Wait( &rcvreq[0], MPI_STATUS_IGNORE );
    MPI_Wait( &rcvreq[1], MPI_STATUS_IGNORE );
    MPI_Wait( &rcvreq[2], MPI_STATUS_IGNORE );
    MPI_Wait( &rcvreq[3], MPI_STATUS_IGNORE );
#endif

    /*
     * Here we would do work that needed to be done after the slice
     * exchange was completed.
     */

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /*
     * Have each process display its portion of the domain, including the
     * extra internal boundary columns.  The MPI_Barrier() and usleep()
     * calls are only necessary to make sure that output is in the proper
     * (i.e. expected) order.
     */
    for ( i = 0; i < num_proc; i++ )
    {
        if ( rank == i )
        {
            printf( "Rank: %d\n", rank );
            showGrid( u, halo_grid.nx, halo_grid.ny );
        }
        MPI_Barrier( comm2d );
        usleep( 10000 ); /* provide time for each process to finish output */
    }

#if defined(GATHER_DATA)
    /*
     * Gather data from all processes, assemble into master grid
     * and display
     */
    gatherData( comm2d, rank, num_proc, u, &orig_grid, &halo_grid, NX, NY );
#endif

#if defined(WRITE_FILE)
    /*
     * Write out single master file containing data from each process
     */
    writeFile( "cart.h5", "/grid", comm2d, u, &orig_grid, &halo_grid, NX, NY );
#endif

    /*
     * Release memory and datatypes and then quit
     */
    free( u );

    MPI_Type_free( &xSlice );
    MPI_Type_free( &ySlice );

    MPI_Finalize();

    return 0;
}