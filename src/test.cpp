#include <utility>
#include <vector>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <algorithm>
#include <iomanip>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctime>
#include <ratio>
#include <chrono>

#pragma omp declare reduction(vec_double_plus : std::vector<double> : \
                              std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<float>())) \
                    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

using vid_t = int64_t;
using weight_t = double;
using vdata_t = int32_t;
using cnt_t = int64_t;

vid_t v_num = 10;
std::vector < std::vector<vid_t> > edge_0; 
std::vector <double> in_degree;

int main(){

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    std::string edge_file = "/workspaces/GEN_LAYPH/dataset/test/test-1.e";
    std::ifstream infile;
    
    edge_0.clear();
    edge_0.resize(v_num);
    in_degree.clear();
    in_degree.resize(v_num);
    infile.open(edge_file);
    vid_t u, v;
    while (infile >> u >> v) // edge from u to v (undirected, no weight)
    {
        edge_0[u].push_back(v);
        if(u != v){
            edge_0[v].push_back(u);
        }
    }
    infile.close();

    omp_set_num_threads(24);
    for(vid_t u = 0; u < v_num; ++u){
        
        #pragma omp parallel for reduction(vec_double_plus : in_degree)
        for(cnt_t k = 0; k < edge_0[u].size(); ++k){
            vid_t v = edge_0[u][k];
            in_degree[v] += 1.0;
        }
    }

    std::vector<cnt_t> prefix_offset; // 存储当前子图 inner vertex 的度数的前缀和
    cnt_t edge_sum = 0; // 存储当前子图边的总数
    prefix_offset.clear();
    prefix_offset.resize(v_num);
    for(vid_t u = 0; u < v_num; ++u){
        edge_sum += edge_0[u].size();
        prefix_offset[u] = edge_sum;
    }

    for(cnt_t i = 1 ; i <= edge_sum; ++i){
        //auto loc = std::lower_bound(prefix_offset.begin(), prefix_offset.end(), i);
        auto loc = std::lower_bound(prefix_offset.begin(), prefix_offset.end(), i) - prefix_offset.begin();
        //std::cout << *loc << std::endl;
        std::cout << loc << " " << edge_0[loc][edge_0[loc].size() - (prefix_offset[loc] - i) - 1] << std::endl;
    }

    for(vid_t u = 0; u < v_num; ++u){
        std::cout << in_degree[u] << std::endl;
        // for(auto v : edge_0[u])
        //     std::cout << u << " " << v << std::endl;
        //std::cout << prefix_offset[u] << std::endl;
    }

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> time_span = duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "It tooks " << time_span.count() << " ms." << std::endl;
    
    return 0;
}