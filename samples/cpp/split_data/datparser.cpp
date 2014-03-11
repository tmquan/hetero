#include "datparser.hpp"

#include <boost/algorithm/string.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <assert.h>
#include <ctype.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace std;

namespace {
vector<string> split_string(const string& str, const string& delimiters)
{
    vector<string> res;

    string split_str = str;
    size_t pos_delim = split_str.find(delimiters);

    while ( pos_delim != string::npos)
    {
        if (pos_delim == 0)
        {
            res.push_back("");
            split_str.erase(0, 1);
        }
        else
        {
            res.push_back(split_str.substr(0, pos_delim));
            split_str.erase(0, pos_delim + 1);
        }

        pos_delim = split_str.find(delimiters);
    }

    res.push_back(split_str);

    return res;
}

string del_space(string name)
{
    while ((name.find_first_of(' ') == 0)  && (name.length() > 0))
        name.erase(0, 1);

    while ((name.find_last_of(' ') == (name.length() - 1)) && (name.length() > 0))
        name.erase(name.end() - 1, name.end());

    return name;
}

}//namespace

void DatFileParser::printContent()
{
	cout << "ObjectFileName:" << objectFileName << endl;
	cout << "Resolution:" << dimx << " " 
						  << dimy << " " 
						  << dimz << endl;
	cout << "Format:" << fileType << endl;
}

DatFileParser::DatFileParser()
{}

DatFileParser::DatFileParser(string fileName)
{
	objectFileName = "";
	dimx = 1;
	dimy = 1;
	dimz = 1;
	
	cout << fileName << endl;
	// fs = new fstream;												
	fs.open(fileName, ios::in);								
	if (!fs.is_open())														
	{																		
		fprintf(stderr, "Cannot open file '%s' in file '%s' at line %i\n",	
		fileName, __FILE__, __LINE__);										
	}	
	
	// Start to read
	string line;
	vector<string> parts;
	
	vector<string>::iterator it;
	string delimiters(" :\t");

    while(getline(fs, line)) // Query the file
	{
		parts.clear();
		// trim the last space
		line.erase(line.find_last_not_of(" \n\r\t")+1);
		// cout << parts.size() << endl;
		// Using boost library to split the string
		boost::split(parts, line, boost::is_any_of(delimiters));
		// cout << parts.size() << endl;
		
		// TODO: Debug
		// for(it=parts.begin(); it != parts.end(); it++)
		// {
			// // cout << parts.at<string>(it) << endl;
			// cout << (*it) << endl;
		// }
		if(parts.at(0) == "ObjectFileName")
		{
			// cout << parts.at(1) << endl;
			// cout << parts[1] << endl;
			objectFileName = parts.at(3);
			// cout << "Debug" << endl;
			// cout << parts[0] << endl;
			// cout << parts[1] << endl;
			// cout << parts[2] << endl;
			// cout << parts[3] << endl;
			// cout << parts[4] << endl;
			objectFileName = fileName.substr(0, fileName.length() - objectFileName.length());
			cout << objectFileName << endl;
			objectFileName += parts.at(3);
		}
		
		
		if(parts.at(0) == "Resolution")
		{
			// cout << parts.at(1) << endl;
			// cout << parts.at(2) << endl;
			// cout << parts.at(3) << endl;
			// cout << parts.at(4) << endl;
			// cout << parts.at(5) << endl;
			// cout << parts.at(6) << endl;
			if(parts.size() == 5)
			{// 2D data
				dimx = getData<int>(parts.at(3));
				dimy = getData<int>(parts.at(4));
			}
			
			if(parts.size() == 6)
			{// 3D data 
				dimx = getData<int>(parts.at(3));
				dimy = getData<int>(parts.at(4));
				dimz = getData<int>(parts.at(5)); //add 1 index
			}
			
		}
		
		if(parts.at(0) == "Format")
		{
			fileType = getData<string>(parts.at(3));
		}
	}
	
	// cout << "ObjectFileName:" << objectFileName << endl;
	// cout << "Resolution:" << dimx << " " 
						  // << dimy << " " 
						  // << dimz << endl;
	// cout << "Format:" << fileType << endl;
	// fs->read(reinterpret_cast<char*>(pData), size);							
	
}

DatFileParser::~DatFileParser()
{
	fs.close();
	// fs->close();															
	// delete fs;												
}