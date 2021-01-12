#pragma once

#include <FlexLexer.h>

#include "../Compiler/INode.h"
#include "gramar.tab.hh"

namespace yy
{

    class Driver {
    private:
        FlexLexer* lexer_ = nullptr;
        ezg::IScope* m_node = nullptr;

    public:
        Driver (FlexLexer* lexer):
                lexer_ (lexer)
        {}
        parser::token_type yylex (parser::semantic_type* yylval) {
            parser::token_type tokenType = static_cast <parser::token_type> (lexer_->yylex ());
            switch (tokenType) {
                case yy::parser::token_type::NUMBER: {
                    yylval->as <double> () = std::stod (lexer_->YYText ());
                    break;
                }
                case yy::parser::token_type::WORD: {
                    std::string word(lexer_->YYText());
                    auto tmp_tt = isKeyWord(word);
                    if (tmp_tt != parser::token_type::ERROR) {
                        tokenType = tmp_tt;
                    }
                    else {
                        yylval->as <std::string> () = word;
                        tokenType = parser::token_type::VARIABLE;
                    }
                    break;
                }
                default: {
                    break;
                }
            }
            return tokenType;
        }

        void setNode(ezg::IScope* sc) { m_node = sc; }

        bool parse () {
            parser parser (this);
            bool result = parser.parse ();
            return !result;
        }

        parser::token_type isKeyWord(const std::string& str)
        {
            parser::token_type res = parser::token_type::ERROR;

            if (str == "while") {
                res = parser::token_type::WHILE;
            }
            else if (str == "if") {
                res = parser::token_type::IF;
            }
            else if (str == "print") {
                res = parser::token_type::PRINT;
            }
            else if (str == "int") {
                res = parser::token_type::TYPE;
            }

            return res;
        }
    };

}//namespace yy