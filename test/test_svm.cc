/*********************************************************************************
*     File Name           :     test_svm_reader.cc
*     Created By          :     yuewu
*     Creation Date       :     [2015-11-11 23:08]
*     Last Modified       :     [2015-11-14 16:43]
*     Description         :     test svm reader and writer
**********************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <vector>

#include "lsol/pario/data_reader.h"
#include "lsol/pario/data_writer.h"
#include "lsol/util/util.h"

using namespace lsol;
using namespace lsol::pario;
using namespace std;

#define CHECK_EQ(x, y) assert(std::abs((x) - (y)) < 1e-6)

int test_svm_reader(const char* path, vector<DataPoint>& dps) {
    printf("load and parse data\n");
    DataReader* reader = DataReader::Create("svm");
    if (reader == nullptr) {
        fprintf(stderr, "create svm reader failed!\n");
        return -1;
    }
    if (reader->Open(path) != Status_OK) {
        return -1;
    }
    int ret = Status_OK;
    for (int i = 0; i < 5; ++i) {
        DataPoint dp;
        printf("parse line %d\n", i);
        ret = reader->Next(dp);
        switch (i) {
            case 0:
                assert(ret == Status_OK);
                //"-1,0.1,-0.1	,1.5e-1	,-1.5e1 ,-1.5e-1 \n"
                CHECK_EQ(dp.label(), -1);
                CHECK_EQ(dp.features(0), 0.1);
                CHECK_EQ(dp.features(1), -0.1);
                CHECK_EQ(dp.features(2), 1.5e-1);
                CHECK_EQ(dp.features(3), -1.5e1);
                CHECK_EQ(dp.features(4), -1.5e-1);
                break;
            case 1:
                assert(ret == Status_OK);
                //"1  ,0.1  ,-0.1	,1.5e-1	,-1.5e1 ,-1.5e-1 \n"
                CHECK_EQ(dp.label(), 1);
                CHECK_EQ(dp.features(0), 0.1);
                CHECK_EQ(dp.features(1), -0.1);
                CHECK_EQ(dp.features(2), 1.5e-1);
                CHECK_EQ(dp.features(3), -1.5e1);
                CHECK_EQ(dp.features(4), -1.5e-1);
                break;
            case 2:
                //"1\t,0.1  :-0.1	,1.5e-1	4:-1.5e1 5:-1.5e-1\n"
                assert(ret == Status_Invalid_Format);
                break;
            case 3:
                assert(ret == Status_OK);
                //"1  ,0.1  ,-0.1	, 1.5e-1	,\t-1.5e1 ,-1.5e-1";
                CHECK_EQ(dp.label(), 1);
                CHECK_EQ(dp.features(0), 0.1);
                CHECK_EQ(dp.features(1), -0.1);
                CHECK_EQ(dp.features(2), 1.5e-1);
                CHECK_EQ(int(dp.indexes(3)), 4);
                CHECK_EQ(dp.features(3), -1.5e1);
                CHECK_EQ(dp.features(4), -1.5e-1);
                break;
            case 4:
                assert(ret == Status_EndOfFile);
                break;
            default:
                break;
        }
        if (ret == Status_OK) dps.push_back(dp);
    }

    delete reader;
    return 0;
}

int test_svm_writer(vector<DataPoint>& dps) {
    const char* out_path = "tmp_test_svm_writer.svm";
    DataWriter* writer = DataWriter::Create("svm");
    if (writer == nullptr) {
        fprintf(stderr, "create svm writer failed!\n");
        return -1;
    }
    if (writer->Open(out_path) != Status_OK) {
        return -1;
    }
    for (const DataPoint& dp : dps) {
        writer->Write(dp);
    }
    delete writer;

    DataReader* reader = DataReader::Create("svm");
    if (reader == nullptr) {
        fprintf(stderr, "create svm reader failed!\n");
        return -1;
    }
    reader->Open(out_path);
    vector<DataPoint> dps2;
    DataPoint dp2;
    while (reader->Next(dp2) == Status_OK) {
        dps2.push_back(dp2);
    }

    delete reader;

    // check if dps and dps2 are the same
    if (dps.size() != dps2.size()) {
        fprintf(stderr, "check svm writer failed!\n");
        return Status_Error;
    }
    for (size_t i = 0; i < dps.size(); ++i) {
        if (dps[i].label() != dps2[i].label()) {
            fprintf(
                stderr,
                "check svm writer failed: label of instance %lu not the same\n",
                i);
            return Status_Error;
        }
        for (size_t j = 0; j < dps[i].indexes().size(); ++j) {
            if (dps[i].indexes(j) != dps2[i].indexes(j)) {
                fprintf(stderr,
                        "check svm writer failed: index %lu of instance %lu "
                        "not the "
                        "same\n",
                        j, i);
                return Status_Error;
            }
            if (dps[i].features(j) != dps2[i].features(j)) {
                fprintf(stderr,
                        "check svm writer failed: feature %lu of instance %lu "
                        "not the "
                        "same\n",
                        j, i);
                return Status_Error;
            }
        }
    }
    delete_file(out_path);
    return Status_OK;
}

int main() {
    const char* test_data =
        "-1  1:0.1  2:-0.1	3:1.5e-1	4:-1.5e1 5:-1.5e-1 \n"
        "1  1:0.1  2:-0.1	3:1.5e-1	4:-1.5e1 5:-1.5e-1 \n"
        "1\t1:0.1  2e-3\t:-0.1	3:1.5e-1	4:-1.5e1 5:-1.5e-1\n"
        "1  1:0.1  2 :-0.1	3: 1.5e-1	4:\t-1.5e1 5:-1.5e-1";

    printf("write test data to disk\n");

    const char* out_path = "tmp_test_svm_reader.svm";
    ofstream out_file(out_path, ios::out);
    if (!out_file) {
        fprintf(stderr, "open %s failed!\n", out_path);
        return -1;
    }
    out_file << test_data;
    out_file.close();

    vector<DataPoint> dps;
    int ret = 0;
    if ((ret = test_svm_reader(out_path, dps)) == 0) {
        printf("%lu features loaded\n", dps.size());
        for (const DataPoint& dp : dps) {
            printf("%d", dp.label());
            for (size_t i = 0; i < dp.indexes().size(); ++i) {
                printf(" %d:%g", dp.indexes(i), dp.features(i));
            }
            printf("\n");
        }
        printf("check svm reader succeed!\n");
    }
    delete_file(out_path, true);

    if ((ret = test_svm_writer(dps)) == Status_OK) {
        printf("check svm writer succeed!\n");
    }
    return ret;
}
