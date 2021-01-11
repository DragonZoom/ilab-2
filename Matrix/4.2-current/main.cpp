#include <iostream>

#include "ParserDriver.h"


#include <set>
#include <vector>
#include <cassert>
#include <algorithm>

// here we can return non-zero if lexing is not done inspite of EOF detected
int yyFlexLexer::yywrap() {
    return 1;
}

int main()
{
    //we shoot a cannon at sparrows
#ifdef DEBUG
    freopen("tests/test3.txt", "r", stdin);
#endif
    FlexLexer* lexer = new yyFlexLexer;
    yy::ParsDriver driver(lexer);
    driver.parse();

    auto circuit  = driver.getData();

#ifdef DEBUG
    std::cout << circuit.dumpStr() << std::endl;
#endif

    circuit.calculateCurrent();

    auto edges = circuit.getData();

    for (auto it = edges.crbegin(); it != edges.crend(); ++it)
    {
        auto cur = *it;
        std:: cout << cur.v1 << " -- " << cur.v2 << ": " << ((cur.current.value() == 0) ? 0 : cur.current.value()) << " A" << std::endl;
    }
}

