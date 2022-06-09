#include <iostream>
#include <vector>

#include <map>
#include <unordered_map>


struct SiaRecord
{
    std::string Name;
    uint64_t Population;
    double Latitude, Longitude;
};

int main()
{
    std::vector<SiaRecord> sia;
    sia.emplace_back("Arash", 50000 , 2,3, 2,5);
    sia.emplace_back("Homa", 50000 ,  2,3, 2,5);
    sia.emplace_back("Siavash", 50000 ,  2,3, 2,5);
    sia.emplace_back("Siamak",  50000 , 2,3, 2,5);
    sia.emplace_back("Rasoul",  50000 , 2,3, 2,5);

    for (const auto& s : sia)
    {
        if (s.Name == "Siamak")
        std::cout << s.Population << std::endl ;
        break;
    }

    return 0 ;
}