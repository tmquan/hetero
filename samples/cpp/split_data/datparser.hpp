#include <map>
#include <new>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>

using namespace std;	
class DatFileParser
{
    public:

    //! the default constructor 
    DatFileParser();
    DatFileParser(string fileName);
	~DatFileParser();
	
	//! print short name, full name, current value and help for all params
    void printContent();
	
	string 	getObjectFileName() {return objectFileName;}
	string 	getFileType() 		{return fileType;}
	int 	getdimx() {return dimx;}
	int 	getdimy() {return dimy;}
	int 	getdimz() {return dimz;}
	
	private:
	fstream fs;
	string objectFileName;
	int dimx;
	int dimy;
	int dimz;
	string fileType;
	
	template<typename _Tp>
    static _Tp getData(const std::string& str)
    {
        _Tp res;
        std::stringstream s1(str);
        s1 >> res;
        return res;
    }
	
	
};