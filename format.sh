#! /bin/bash
for ext in *.h *.cc
do
    src_list=$(find . -name $ext)
    for src in ${src_list}
    do
        echo 'process' $src
        vim -c ":ClangFormat" -c ":x" $src
    done
done
