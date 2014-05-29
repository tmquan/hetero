#include <map>
#include <new>
#include <string>
#include <vector>
#include <sstream>
/*!
"\nThe CommandLineParser class is designed for command line arguments parsing\n"
           "Keys map: \n"
           "Before you start to work with CommandLineParser you have to create a map for keys.\n"
           "    It will look like this\n"
           "    const char* keys =\n"
           "    {\n"
           "        {    s|  string|  123asd |string parameter}\n"
           "        {    d|  digit |  100    |digit parameter }\n"
           "        {    c|noCamera|false    |without camera  }\n"
           "        {    1|        |some text|help            }\n"
           "        {    2|        |333      |another help    }\n"
           "    };\n"
           "Usage syntax: \n"
           "    \"{\" - start of parameter string.\n"
           "    \"}\" - end of parameter string\n"
           "    \"|\" - separator between short name, full name, default value and help\n"
           "Supported syntax: \n"
           "    --key1=arg1  <If a key with '--' must has an argument\n"
           "                  you have to assign it through '=' sign.> \n"
           "<If the key with '--' doesn't have any argument, it means that it is a bool key>\n"
           "    -key2=arg2   <If a key with '-' must has an argument \n"
           "                  you have to assign it through '=' sign.> \n"
           "If the key with '-' doesn't have any argument, it means that it is a bool key\n"
           "    key3                 <This key can't has any parameter> \n"
           "Usage: \n"
           "      Imagine that the input parameters are next:\n"
           "                -s=string_value --digit=250 --noCamera lena.jpg 10000\n"
           "    CommandLineParser parser(argc, argv, keys) - create a parser object\n"
           "    parser.get<string>(\"s\" or \"string\") will return you first parameter value\n"
           "    parser.get<string>(\"s\", false or \"string\", false) will return you first parameter value\n"
           "                                                                without spaces in end and begin\n"
           "    parser.get<int>(\"d\" or \"digit\") will return you second parameter value.\n"
           "                    It also works with 'unsigned int', 'double', and 'float' types>\n"
           "    parser.get<bool>(\"c\" or \"noCamera\") will return you true .\n"
           "                                If you enter this key in commandline>\n"
           "                                It return you false otherwise.\n"
           "    parser.get<string>(\"1\") will return you the first argument without parameter (lena.jpg) \n"
           "    parser.get<int>(\"2\") will return you the second argument without parameter (10000)\n"
           "                          It also works with 'unsigned int', 'double', and 'float' types \n"
*/

class CommandLineParser
{
    public:

    //! the default constructor
      CommandLineParser(int argc, const char* const argv[], const char* key_map);

    //! get parameter, you can choose: delete spaces in end and begin or not
    template<typename _Tp>
    _Tp get(const std::string& name, bool space_delete=true)
    {
        if (!has(name))
        {
            return _Tp();
        }
        std::string str = getString(name);
        return analyzeValue<_Tp>(str, space_delete);
    }

    //! print short name, full name, current value and help for all params
    void printParams();

    protected:
    std::map<std::string, std::vector<std::string> > data;
    std::string getString(const std::string& name);

    bool has(const std::string& keys);

    template<typename _Tp>
    _Tp analyzeValue(const std::string& str, bool space_delete=false);

    template<typename _Tp>
    static _Tp getData(const std::string& str)
    {
        _Tp res;
        std::stringstream s1(str);
        s1 >> res;
        return res;
    }

    template<typename _Tp>
     _Tp fromStringNumber(const std::string& str);//the default conversion function for numbers

    };

template<>  
bool CommandLineParser::get<bool>(const std::string& name, bool space_delete);

template<>  
std::string CommandLineParser::analyzeValue<std::string>(const std::string& str, bool space_delete);

template<>  
int CommandLineParser::analyzeValue<int>(const std::string& str, bool space_delete);

template<>  
unsigned int CommandLineParser::analyzeValue<unsigned int>(const std::string& str, bool space_delete);


template<>  
float CommandLineParser::analyzeValue<float>(const std::string& str, bool space_delete);

template<>  
double CommandLineParser::analyzeValue<double>(const std::string& str, bool space_delete);