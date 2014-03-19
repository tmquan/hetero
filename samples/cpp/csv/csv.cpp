#include <fstream>
#include <sstream>

int main(int argc, char **argv)
{
    const char *filename = argv[1];
    const int M = 1000000;
    const char str_delimiter[3] = { ',', delimiter, '\0' };
    // FILE* file = 0;
    // CvMemStorage* storage;
    // CvSeq* seq;
    // char *ptr;
    // float* el_ptr;
    // CvSeqReader reader;
    int cols_count = 0;
    uchar *var_types_ptr = 0;

    clear();

    file = fopen( filename, "rt" );

    if( !file )
        return -1;

    // read the first line and determine the number of variables
    std::vector<char> _buf(M);
    char* buf = &_buf[0];
    if( !fgets_chomp( buf, M, file ))
    {
        fclose(file);
        return -1;
    }

    ptr = buf;
    while( *ptr == ' ' )
        ptr++;
    for( ; *ptr != '\0'; )
    {
        if(*ptr == delimiter || *ptr == ' ')
        {
            cols_count++;
            ptr++;
            while( *ptr == ' ' ) ptr++;
        }
        else
            ptr++;
    }

    cols_count++;

    if ( cols_count == 0)
    {
        fclose(file);
        return -1;
    }

    // // create temporary memory storage to store the whole database
    // el_ptr = new float[cols_count];
    // storage = cvCreateMemStorage();
    // seq = cvCreateSeq( 0, sizeof(*seq), cols_count*sizeof(float), storage );

    // var_types = cvCreateMat( 1, cols_count, CV_8U );
    // cvZero( var_types );
    // var_types_ptr = var_types->data.ptr;

    // for(;;)
    // {
        // char *token = NULL;
        // int type;
        // token = strtok(buf, str_delimiter);
        // if (!token)
            // break;
        // for (int i = 0; i < cols_count-1; i++)
        // {
            // str_to_flt_elem( token, el_ptr[i], type);
            // var_types_ptr[i] |= type;
            // token = strtok(NULL, str_delimiter);
            // if (!token)
            // {
                // fclose(file);
                // return -1;
            // }
        // }
        // str_to_flt_elem( token, el_ptr[cols_count-1], type);
        // var_types_ptr[cols_count-1] |= type;
        // cvSeqPush( seq, el_ptr );
        // if( !fgets_chomp( buf, M, file ) )
            // break;
    // }
    // fclose(file);

    // values = cvCreateMat( seq->total, cols_count, CV_32FC1 );
    // missing = cvCreateMat( seq->total, cols_count, CV_8U );
    // var_idx_mask = cvCreateMat( 1, values->cols, CV_8UC1 );
    // cvSet( var_idx_mask, cvRealScalar(1) );
    // train_sample_count = seq->total;

    // cvStartReadSeq( seq, &reader );
    // for(int i = 0; i < seq->total; i++ )
    // {
        // const float* sdata = (float*)reader.ptr;
        // float* ddata = values->data.fl + cols_count*i;
        // uchar* dm = missing->data.ptr + cols_count*i;

        // for( int j = 0; j < cols_count; j++ )
        // {
            // ddata[j] = sdata[j];
            // dm[j] = ( fabs( MISS_VAL - sdata[j] ) <= FLT_EPSILON );
        // }
        // CV_NEXT_SEQ_ELEM( seq->elem_size, reader );
    // }

    // if ( cvNorm( missing, 0, CV_L1 ) <= FLT_EPSILON )
        // cvReleaseMat( &missing );

    // cvReleaseMemStorage( &storage );
    // delete []el_ptr;
    // return 0;
}