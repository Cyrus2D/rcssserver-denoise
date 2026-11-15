// Compile with the actual soccerwindow2 parser and -fno-access-control.
// parseImpl bypasses only the UI holder, not the viewer packet grammar.
#include "debug_client_parser.h"
#include <fstream>
#include <iostream>
#include <cassert>
#include <cstdio>
#include <string>
int main(int argc,char **argv) {
    assert(argc>=2);size_t packets=0,max_size=0;
    for(int i=1;i<argc;++i) {
        std::ifstream file(argv[i]);assert(file.good());std::string line;
        while(std::getline(file,line)) {
            if(line.rfind("%% step",0)==0||line.empty())continue;
            if(line.rfind("%% debug [",0)==0){line=line.substr(10);line.pop_back();}
            int version=0,n=0;
            assert(std::sscanf(line.c_str()," ( ( debug ( format-version %d ) ) %n",&version,&n)==1&&version==5);
            DebugClientParser parser;DebugViewData::Ptr view(new DebugViewData);
            assert(parser.parseImpl(line.c_str()+n,view));
            assert(parser.M_unum>=1&&parser.M_unum<=11);
            max_size=std::max(max_size,line.size()+1);++packets;
        }
    }
    assert(packets>0);
    std::cout<<"PASS: soccerwindow2 parser accepted "<<packets<<" packets; max "<<max_size<<" bytes including NUL\n";
}
