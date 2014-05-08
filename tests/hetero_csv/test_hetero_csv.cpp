#include <iostream>
#include <hetero_csv.hpp>

using namespace std;

int main(int argc, char **argv)
{
    
    vector<string> fields;
    vector<string> values;
    
    read_csv(argv[1], &fields, &values);
    
    for(int i=0; i<fields.size(); i++)
    {
        cout << fields[i] << "\t" << values[i] << endl;
    }
    return 0;
}