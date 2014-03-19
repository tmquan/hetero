#include <iostream>
#include <hetero_csv.hpp>
#include <sstream>      // std::istringstream

using namespace std;

int main(int argc, char **argv)
{
    
    vector<string> fields;
    vector<string> values;
    
    read_csv(argv[1], &fields, &values);
    
    // for(int i=0; i<fields.size(); i++)
    // {
        // cout << fields[i] << "\t" << values[i] << endl;
    // }
    
    string objectFileName;
    int dimx;
    int dimy;
    int dimz;
    string type;
    
    for(int i=0; i<fields.size(); i++)
    {
        if(fields[i] == "objectFileName")   objectFileName  = values[i];
        if(fields[i] == "format")           type  = values[i];
        // if(fields[i] == "dimx")             dimx            = (int)atoi(values[i]);
        // if(fields[i] == "dimy")             dimy            = (int)atoi(values[i]);
        // if(fields[i] == "dimz")             dimz            = (int)atoi(values[i]);
        if(fields[i] == "dimx")             istringstream(values[i]) >> dimx;
        if(fields[i] == "dimy")             istringstream(values[i]) >> dimy;
        if(fields[i] == "dimz")             istringstream(values[i]) >> dimz;
    }
    
    cout << objectFileName << endl;
    cout << type << endl;
    cout << dimx << endl;
    cout << dimy << endl;
    cout << dimz << endl;
    
    return 0;
}