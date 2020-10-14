#include <iostream>
#include <list>
#include <vector>
#include <map>
#include <set>
#include <fstream>

//#define DEBUGG

#include "Octree/Octree.h"
#include "tests/test_gen.h"

extern size_t COUNT_TT_INTERSEC;



std::vector< la::Triangle > getData(std::istream& _source)
{
    int n = 0;
    _source >> n;
    std::vector< la::Triangle > data;
    data.reserve(n);
    for (int i = 0; i < n; i++)
    {
        la::Vector3f a, b, c;
        _source >> a.x >> a.y >> a.z
            >> b.x >> b.y >> b.z
            >> c.x >> c.y >> c.z;
        data.push_back(la::Triangle(a, b, c));
    }

    return data;
}



template <typename Val_, typename Key_ = int>
struct ValId
{
    Val_ val;
    Key_ id;

    la::Square getArea() const noexcept { return val.getArea(); }
};
bool intersec(const ValId<la::Triangle, size_t>& _lhs, const ValId<la::Triangle, size_t>& _rhs)
{
    return la::intersec(_lhs.val, _rhs.val);
}


void intersectionN_N(std::vector< la::Triangle > data)
{
    const size_t n = data.size();

    COUNT_TT_INTERSEC = 0u;

    //std::cout << data[26].first << ' ' << data[12].first << std::endl;

    std::vector< std::pair< ValId< la::Triangle, size_t >, ValId< la::Triangle, size_t > > > result_pair;
    std::set<size_t> res_id;

    size_t counter_cmp = 0;
    for (size_t i = 0; i < n; i++)
    {
        auto& lhs = data.at(i);

        for (size_t k = i + 1; k < n; k++)
        {
            auto& rhs = data.at(k);

            if (la::intersec(lhs, rhs))
            {
                result_pair.push_back(std::make_pair(ValId<la::Triangle, size_t>{lhs, i},
                    ValId<la::Triangle, size_t>{rhs, k}));
                res_id.insert(i);
                res_id.insert(k);
            }
            counter_cmp++;
        }
    }

    std::cout
        << "\nN*N\n"
        << "N        = " << n << '\n'
        << "N * N    = " << n * n << '\n'
        << "num cmp  = " << COUNT_TT_INTERSEC << '\n';

#ifdef DEBUGG

    for (const auto& r : result_pair)
    {
        std::cout << '(' << r.first.id << ", " << r.second.id << ") ";
    }

#else

    for (const auto& id : res_id)
    {
        std::cout << id << ' ';
    }

#endif //DEBUGG

    std::cout << std::endl;
}


#define MIN(a, b, c) \
std::min(a, std::min(b, c))

#define MAX(a, b, c) \
std::max(a, std::max(b, c))

void intersectionOctree(std::vector< la::Triangle > data)
{
    using std::min;
    using std::max;

    COUNT_TT_INTERSEC = 0u;

    const size_t n = data.size();

    if (!data.empty())
    {
        la::Vector3f a = data.at(0).getA(), b = data.at(0).getB();

        for (size_t i = 0, n = data.size(); i < n; ++i)
        {
            const la::Triangle tr = data.at(i);
            a.x = min(a.x, MIN(tr.getA().x, tr.getB().x, tr.getC().x));
            a.y = min(a.y, MIN(tr.getA().y, tr.getB().y, tr.getC().y));
            a.z = min(a.z, MIN(tr.getA().z, tr.getB().z, tr.getC().z));
            b.x = max(b.x, MAX(tr.getA().x, tr.getB().x, tr.getC().x));
            b.y = max(b.y, MAX(tr.getA().y, tr.getB().y, tr.getC().y));
            b.z = max(b.z, MAX(tr.getA().z, tr.getB().z, tr.getC().z));
        }

        la::Octree< ValId< la::Triangle, size_t > > tree(la::Square(a, b));

        for (size_t i = 0, n = data.size(); i < n; ++i)
        {
            tree.add({ data.at(i), i });
        }

        tree.msplit();
        auto res = tree.getIntersections();

        std::set<int> id_res;
        for (const auto& p : res)
        {
            id_res.insert(p.first.id);
            id_res.insert(p.second.id);
        }

        std::cout << '\n';
        std::cout << tree.dumpStr() << std::endl;
        std::cout << "num cmp       = " << COUNT_TT_INTERSEC << std::endl;

#ifdef DEBUGG
        for (const auto& r : res)
        {
            std::cout << '(' << r.first.id << ", " << r.second.id << ") ";
        }
#else
        for (const int& k : id_res)
        {
            std::cout << k << ' ';
        }
#endif // DEBUGG

        COUNT_TT_INTERSEC = 0u;

        std::cout << std::endl;
    }

}