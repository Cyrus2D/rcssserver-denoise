#!/usr/bin/env python3
"""Link a white-box fixture against the actual CMake-built server objects."""
import argparse,shlex,subprocess
from pathlib import Path
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("build",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args();build=a.build.resolve();repo=build.parent
    link=shlex.split((build/"src/CMakeFiles/RCSSServer.dir/link.txt").read_text())
    objects=[];flags=[];skip=False
    for arg in link[1:]:
        if skip:skip=False;continue
        if arg=="-o":skip=True;continue
        if arg.endswith("main.cpp.o")or arg in ["-O3","-DNDEBUG"]:continue
        if arg.endswith(".o")or arg.endswith(".so")or ".so."in arg:
            objects.append(str((build/"src"/arg).resolve())if not Path(arg).is_absolute()else arg)
        elif arg.startswith("-Wl,"):flags.append(arg)
    source=Path(__file__).with_name("command_order.cpp").resolve()
    command=["g++","-std=c++17","-O2","-fno-access-control","-DHAVE_CONFIG_H",
             "-I"+str(repo),"-I"+str(repo/"src"),"-I"+str(build),"-I"+str(build/"src"),
             str(source),*objects,*flags,"-o",str(a.output.resolve())]
    subprocess.run(command,check=True)
