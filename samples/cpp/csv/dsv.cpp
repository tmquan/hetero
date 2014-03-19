#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

using namespace std;

int main(int argc, char **argv)
{
    // float data[38][27];
    std::ifstream file(argv[1]);
    vector<std::string> fields;
    vector<std::string> values;
    // for(int row = 0; row < 38; ++row)
    // {
        // std::string line;
        // std::getline(file, line);
        // if ( !file.good() )
            // break;

        // std::stringstream iss(line);

        // for (int col = 0; col < 27; ++col)
        // {
            // std::string val;
            // std::getline(iss, val, ',');
            // if ( !iss.good() )
                // break;

            // std::stringstream convertor(val);
            // convertor >> data[row][col];
        // }
    // }
    // std::string file = argv[1];
    std::string line;
    std::string delimiter = ",";
    size_t pos = 0;
    std::string token;
    std::getline(file, line);
    while ((pos = line.find(delimiter)) != std::string::npos) 
    {
    
        token = line.substr(0, pos);
        std::cout << token << std::endl;
        line.erase(0, pos + delimiter.length());
    }
    std::cout << line << std::endl;
    
    // if ( !file.good() )
        // break;

    std::stringstream iss(line);
    return 0;
}