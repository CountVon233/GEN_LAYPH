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

vid_t v_num = 0; // node number 
vdata_t max_comm = 0; // louvain生成的聚类数
vdata_t valid_comm_cnt = 0; // 有优化效果的聚类数

std::vector <vdata_t> vdata; //记录louvain生层的每个点的聚类号
std::vector < std::vector<vid_t> > edge_0; // node with  its edge in initial graph
std::vector <cnt_t> inner_edge_comm; // 记录每个聚类的内部边数
std::vector< std::vector<vid_t> > comm_bound_node; // 每个聚类的边界点
std::vector< std::vector<vid_t> > comm_inner_node; // 每个聚类的内部点
std::vector <bool> is_bound; // 记录每个点是否是边界点
std::map<vdata_t, vdata_t> valid_comm_map;

std::vector< cnt_t> short_cnt_num;

void Init(){
    v_num = vdata.size();
    inner_edge_comm.resize(max_comm, 0);
    comm_bound_node.clear();
    comm_inner_node.clear();
    comm_bound_node.resize(max_comm);
    comm_inner_node.resize(max_comm);

    edge_0.clear();
    edge_0.resize(v_num);

    is_bound.resize(v_num, false);
    
    valid_comm_map.clear();
    short_cnt_num.resize(max_comm, 0);
}


int main(){
    
    std::ofstream logfile("/workspaces/GEN_LAYPH/log/preprocess_log");

    logfile << " vfile input " << std::endl;

    std::string louvain_vertex_result = "/workspaces/GEN_LAYPH/dataset/test/test_node2comm_level";
    //std::string louvain_vertex_result = "/workspaces/GEN_LAYPH/dataset/LiveJournal/LiveJournal_node2comm_level";

    std::ifstream infile(louvain_vertex_result);
    vid_t v;
    vdata_t vd;
    vdata.clear();
    while (infile >> v >> vd) // vid and vd must start from 0
    {
        vdata.push_back(vd); //确定每个点及其所在聚类
        max_comm = std::max(max_comm, vd); //得到聚类范围
    }
    max_comm++;
    infile.close();
    
    logfile << " initial variables " << std::endl;

    // 初始化
    Init();

    // std::cout << v_num << std::endl;
    // for(vid_t i = 0; i < v_num; ++i){
    //     std::cout << i << " " << vdata[i] << std::endl;
    // }

    logfile << " efile input " << std::endl;

    // 输入边
    std::string edge_file = "/workspaces/GEN_LAYPH/dataset/test/test-1.e";
    //std::string edge_file = "/workspaces/GEN_LAYPH/dataset/LiveJournal/LiveJournal.e";
    
    infile.open(edge_file);
    vid_t u;
    while (infile >> u >> v) // edge from u to v (undirected, no weight)
    {

        edge_0[u].push_back(v);
        if(vdata[u] == vdata[v]) // 这条边是聚类的内部边
            inner_edge_comm[vdata[u]]++;
        else
            is_bound[u] = true; // 点u是所在聚类的边界点
        if(u != v){
            edge_0[v].push_back(u);
            if(vdata[v] == vdata[u]) // 这条边是聚类的内部边
                inner_edge_comm[vdata[v]]++;
            else
                is_bound[v] = true; // 点v是所在聚类的边界点
        }
    }
    infile.close();

    // for(vid_t i = 0; i < v_num; i++){
    //     for(cnt_t j = 0; j < edge_0[i].size(); ++j){
    //         std::cout << i << " " << edge_0[i][j] << std::endl;
    //     }
    // }

    logfile << " set bound node " << std::endl;

    for(vid_t i = 0; i < v_num; ++i){
        if(is_bound[i])
            comm_bound_node[vdata[i]].push_back(i); // 是边界点就加入所在聚类的边界点集
        else
            comm_inner_node[vdata[i]].push_back(i); // 不是边界点就加入所在聚类的内部点集
    }

    cnt_t bound_node_sum = 0;

    for(cnt_t i = 0; i < max_comm; ++i){
        //std::cout << "for comm " << i  << " : " << std::endl;
       // std::cout << "bound node are : ";
        // for(vid_t v : comm_bound_node[i]){
        //     //std::cout << v << " ";
        // }
        bound_node_sum += comm_bound_node[i].size();
        // std::cout << std::endl << " inner node are : ";
        // for(vid_t v : comm_inner_node[i]){
        //     std::cout << v << " ";
        // }
        // std::cout << std::endl;
    }
    std::cout << bound_node_sum << std::endl;
    

    omp_set_num_threads(24);

    logfile << "count short cuts" << std::endl;

    std::cout << "count short cuts begin " << std::endl;

    std::vector<double> reach(v_num, 0.0);
    //std::vector<double> residue(v_num, 0.0);
    std::vector<double> next_reach(v_num, 0.0);
    //std::vector<double> value(v_num, 0.0); 
    std::vector<cnt_t> prefix_offset{}; // 存储当前子图 inner vertex 的度数的前缀和
    cnt_t edge_sum; // 存储当前子图边的总数
    // std::vector<bool> is_active(v_num, false); // 标记节点是否已经加入active 队列
    // std::queue<vid_t> active_v; // active vertex queue

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    //#pragma omp parallel for private(residue, value, is_active, active_v)
    for(vdata_t i = 0; i < max_comm; ++i){ // 考察每个子图
        
        #pragma omp parallel for
        for(cnt_t k = 0; k < 24; ++k){
            if(i == 0 && k == 0)
                logfile << "number of threads is " << omp_get_num_threads() << std::endl;
        }

        //residue.resize(v_num);
        //value.resize(v_num);
        //is_active.resize(v_num);
        // 对每一个子图，初始化 prefix_offset 和 edge_sum
        prefix_offset.resize(comm_inner_node[i].size(), 0);
        edge_sum = 0;
        for(cnt_t j = 0; j < comm_inner_node[i].size(); ++j){
            vid_t u = comm_inner_node[i][j];
            edge_sum += edge_0[u].size();
            prefix_offset[j] = edge_sum;
        }
        
        for(cnt_t j = 0; j < comm_bound_node[i].size(); ++j){ // 考虑当前聚类的每个边界点
            
            vid_t u = comm_bound_node[i][j]; // u为当前考虑的边界点

            //logfile << "comm id is " << i << " bound node is " << u << std::endl;

            std::fill(reach.begin(), reach.end(), 0.0);
            std::fill(next_reach.begin(), next_reach.end(), 0.0);
            // std::fill(value.begin(), value.end(), 0.0);
            //std::fill(is_active.begin(), is_active.end(), false);
            reach[u] = 1.0; // send u a unit message
            //active_v.push(u); // 将 u 加入 active 队列
            //is_active[u] = true; //将 u 标记为已激活
            cnt_t active_cnt;
            
            #pragma omp parallel for reduction(vec_double_plus : next_reach)
            for(cnt_t k = 0; k < edge_0[u].size(); ++k){
               vid_t v = edge_0[u][k];
               next_reach[v] = 1.0;
            }

            active_cnt = 0;
            #pragma omp parallel for reduction(+ : active_cnt)
            for(cnt_t j = 0; j < comm_inner_node[i].size(); ++j){
                vid_t v = comm_inner_node[i][j];
                if(reach[v] == 0.0 && next_reach[v] > 0.0){
                    reach[v] = 1.0;
                    active_cnt++;
                    //std::cout << " vertex " << v << " is active " << j << std::endl;
                }
                next_reach[v] = 0.0;
            }

            //std::cout << "active vecter number is " << active_cnt << std::endl;

            #pragma omp parallel for
            for(cnt_t j = 0; j < comm_bound_node[i].size(); ++j){
                vid_t v = comm_bound_node[i][j];
                if(reach[v] == 0.0 && next_reach[v] > 0.0)
                    reach[v] = 1.0;
                next_reach[v] = 0.0;   
            }
            reach[u] = 2.0;
            
            while(active_cnt != 0){
                
                #pragma omp parallel for reduction(vec_double_plus : next_reach)
                for(cnt_t k = 1; k <= edge_sum; ++k){ // 用二分查找得到第 k 条边的源点和目的点
                    auto loc = std::lower_bound(prefix_offset.begin(), prefix_offset.end(), k) - prefix_offset.begin();
                    vid_t s = comm_inner_node[i][loc];
                    vid_t t = edge_0[s][edge_0[s].size() - (prefix_offset[loc] - k) - 1];
                    if(reach[s] == 1.0)
                        next_reach[t] = 1.0;
                }

                active_cnt = 0;
                #pragma omp parallel for reduction(+ : active_cnt)
                for(cnt_t j = 0; j < comm_inner_node[i].size(); ++j){
                    vid_t v = comm_inner_node[i][j];
                    if(reach[v] == 1.0){
                        reach[v] = 2.0;
                    }else if (reach[v] == 0.0 && next_reach[v] > 0.0){
                        reach[v] = 1.0;
                        active_cnt++;
                        //std::cout << " vertex " << v << " is active " << std::endl;
                    }
                    next_reach[v] = 0.0;
                }

                //std::cout << "active vecter number is " << active_cnt << std::endl;

                #pragma omp parallel for
                for(cnt_t j = 0; j < comm_bound_node[i].size(); ++j){
                    vid_t v = comm_bound_node[i][j];
                    if(reach[v] == 0.0 && next_reach[v] > 0.0)
                        reach[v] = 1.0;
                    next_reach[v] = 0.0; 
                }
            }
            
            cnt_t short_cnt = 0;
            #pragma omp parallel for reduction(+ : short_cnt)
            for(cnt_t k = 0; k < comm_bound_node[i].size(); ++k){
                vid_t v = comm_bound_node[i][k];
                if(reach[v] > 0.0){
                    short_cnt++;
                    //logfile << "vertex " << u << " has a short cut to " << v << std::endl;
                }
            }
            short_cnt_num[i] = short_cnt;

            //std::cout << "------------------------------------------" << std::endl;
        }
    }

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

    logfile << " set valid commuinty " << std::endl;
    
    for(vdata_t i = 0; i < max_comm; ++i){
        
        //std::cout << "short cuts num of low graph " << i << " is " << short_cnt_num[i] << std::endl;
        //std::cout << "inner edges num of low graph " << i << " is " << inner_edge_comm[i] << std::endl;

        if(inner_edge_comm[i] >= short_cnt_num[i]){ // 有优化效果
            valid_comm_map[i] = valid_comm_cnt; //标记为有优化效果
            valid_comm_cnt++;
        } else{
            valid_comm_map[i] = -1; // 标记为无优化效果
        }
    }

    logfile << "output vertex with valid comm " << std::endl;
    
    std::string two_layer_vertex_map = "/workspaces/GEN_LAYPH/dataset/test/test-two-layer.v";
    //std::string two_layer_vertex_map = "/workspaces/GEN_LAYPH/dataset/LiveJournal/LiveJournal-two-layer.v";

    std::ofstream outfile(two_layer_vertex_map);
    for(vid_t i = 0; i < v_num; ++i){
        outfile << i << " " << valid_comm_map[vdata[i]] << std::endl;
    }

    outfile.close();

    std::string bound_node_on_upgraph = "/workspaces/GEN_LAYPH/dataset/test/test-upgraph-node.v";
    //std::string bound_node_on_upgraph = "/workspaces/GEN_LAYPH/dataset/LiveJournal/LiveJournal-upgraph-node.v";
    
    outfile.open(bound_node_on_upgraph);
    for(vid_t i = 0; i < v_num; ++i){
        if(is_bound[i])
            outfile << i << " " << valid_comm_map[vdata[i]] << std::endl;
    }

    outfile.close();

    logfile.close();

    std::chrono::duration<double> time_span = duration_cast<std::chrono::seconds>(end - start);

    std::cout << "It tooks " << time_span.count() << " s." << std::endl;
    
    return 0;
}