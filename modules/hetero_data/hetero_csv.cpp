#include "hetero_csv.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdlib.h>     // exit, EXIT_FAILURE 

using namespace std;


void read_csv(string filename, vector<string> *fields, vector<string> *values)
{
    // Open the file
    ifstream infile(filename.c_str());
    bool isField = true;
    
    //Read lines from the stream
    while (infile)
    {
        string s;
        if (!getline( infile, s )) break;

        istringstream ss( s );
        while (ss)
        {
            string s;
            // tokenize this stream
            if (!getline( ss, s, ',' )) break;
            if(isField) fields->push_back( s );
            
            else values->push_back( s );
        }
        isField = false;
    }
    if (!infile.eof())
    {
        cerr << "Cannot open " << filename << endl;
        exit (EXIT_FAILURE);
    }
}