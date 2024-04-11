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

using vid_t = int64_t;
using weight_t = double;
using vdata_t = int32_t;
using cnt_t = int64_t;

vid_t v_num = 0; // node number 
vdata_t max_comm = 0; // louvain生成的聚类数
//vdata_t valid_comm_cnt = 0; // 有优化效果的聚类数
double alpha = 0.15;
double thr = 1e-8;
std::vector <vdata_t> vdata; //记录louvain生层的每个点的聚类号
std::vector < std::vector<vid_t> > edge_0; // node with  its edge in initial graph
std::vector <cnt_t> inner_edge_comm; // 记录每个聚类的内部边数
std::vector< std::vector<vid_t> > comm_bound_node; // 每个聚类的边界点
std::vector< std::vector<vid_t> > comm_inner_node; // 每个聚类的内部点
std::vector <bool> is_bound; // 记录每个点是否是边界点
//std::map<vdata_t, vdata_t> valid_comm_map;

//std::vector< std::vector< vid_t > > bound_edge;
//std::vector< std::map< vid_t, weight_t > > bound_edge_weight;

//std::vector< std::vector< std::vector<vid_t> > > low_graph;
std::vector<std::map<vid_t, std::vector<vid_t>> > low_graph;
std::vector< std::vector < std::pair<vid_t, weight_t> > > up_graph;
std::vector< std::map< vid_t, std::vector< std::pair<vid_t, weight_t> > > >  short_cuts;
std::vector< cnt_t> short_cnt_num;
//std::vector< std::vector< std::vector< std::pair<vid_t, weight_t> > > > assign_graph;
std::vector< std::map< vid_t, std::vector< std::pair<vid_t, weight_t> > > >  assign_graph;

bool cmp (std::pair<vid_t, weight_t> a, std::pair<vid_t, weight_t> b){
    return a.first < b.first;
}

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

    low_graph.resize(max_comm);
    assign_graph.resize(max_comm);
    short_cuts.resize(max_comm);
    short_cnt_num.resize(max_comm, 0);
    for(vdata_t i = 0; i < max_comm; ++i){
        low_graph[i].clear();
        assign_graph[i].clear();
        short_cuts[i].clear();
    }
    up_graph.resize(v_num);
}


int main(){
    
    std::ofstream logfile("/workspaces/GEN_LAYPH/log/log_file");
    //输入点和点所在的聚类号

    logfile << " vfile input " << std::endl;

    //std::string two_layer_vertex_map = "/workspaces/GEN_LAYPH/dataset/test/test-two-layer.v";
    //std::string two_layer_vertex_map = "/workspaces/GEN_LAYPH/dataset/LiveJournal/LiveJournal-two-layer.v";
    std::string two_layer_vertex_map = "/workspaces/GEN_LAYPH/dataset/uk2002/uk2002-two-layer.v";
    //std::string two_layer_vertex_map = "/workspaces/GEN_LAYPH/dataset/uk2005/uk2005-two-layer.v";

    std::ifstream infile(two_layer_vertex_map);
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
    
    /*std::cout << v_num << std::endl;
    for(vid_t i = 0; i < v_num; ++i){
        std::cout << i << " " << vdata[i] << std::endl;
    }*/

    logfile << " initialization " << std::endl;

    // 初始化
    Init();

    logfile << " efile input " << std::endl;

    // 输入边
    //std::string edge_file = "/workspaces/GEN_LAYPH/dataset/test/test-1.e";
    //std::string edge_file = "/workspaces/GEN_LAYPH/dataset/LiveJournal/LiveJournal.e";
    std::string edge_file = "/workspaces/GEN_LAYPH/dataset/uk2002/uk2002.e";
    //std::string edge_file = "/workspaces/GEN_LAYPH/dataset/uk2005/uk2005.e";

    infile.open(edge_file);
    vid_t u;
    while (infile >> u >> v) // edge from u to v (undirected, no weight)
    {

        edge_0[u].push_back(v);
        
        if(vdata[u] != -1){
            if(vdata[u] == vdata[v]) // 这条边是聚类的内部边
                inner_edge_comm[vdata[u]]++;
            else
                is_bound[u] = true; // 点u是所在聚类的边界点
        }

        if(u != v){
            edge_0[v].push_back(u);
            if(vdata[v] != -1){
                if(vdata[v] == vdata[u]) // 这条边是聚类的内部边
                    inner_edge_comm[vdata[v]]++;
                else
                    is_bound[v] = true; // 点v是所在聚类的边界点
            }
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
        if(vdata[i] != -1){
            if(is_bound[i])
                comm_bound_node[vdata[i]].push_back(i); // 是边界点就加入所在聚类的边界点集
            else
                comm_inner_node[vdata[i]].push_back(i); // 不是边界点就加入所在聚类的内部点集
        }
        
    }

    logfile << " assign inital edges (up graph edges and inner edges) " << std::endl;

    for(vid_t i = 0; i < v_num; ++i){
        if(vdata[i] != -1){
            for(cnt_t j = 0; j < edge_0[i].size(); ++j){
                vid_t k = edge_0[i][j]; 
                if(vdata[i] != vdata[k]){ // 点 i 到 k 的边是边界边 
                    up_graph[i].push_back( std::make_pair(k, (1 - alpha) / (double)edge_0[i].size()) );
                } else { // 点 i 到 k 的边是内部边
                    if(is_bound[i] == false) // 从边界点指向内部点的内部边不保存到下层图
                        low_graph[vdata[i]][i].push_back(k);  // 将内部边加入对应子图内
                }
            }
        }
        else {
            for(cnt_t j = 0; j < edge_0[i].size(); ++j){
                vid_t k = edge_0[i][j];
                up_graph[i].push_back( std::make_pair(k, (1 - alpha) / (double)edge_0[i].size()) );
            }
        }
    }

    omp_set_num_threads(20);

    logfile << "compute short cuts" << std::endl;

    std::cout << "compute short cuts begin " << std::endl;

    std::vector<double> residue(v_num, 0.0);
    std::vector<double> value(v_num, 0.0); 
    std::vector<bool> is_active(v_num, false); // 标记节点是否已经加入active 队列
    std::queue<vid_t> active_v; // active vertex queue

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for private(residue, value, is_active, active_v)
    for(vdata_t i = 0; i < max_comm; ++i){ // 对每个聚类作并行
        
        if(i == 0)
            logfile << "number of threads is " << omp_get_num_threads() << std::endl;

        //logfile << "comm id is " << i << std::endl;

        residue.resize(v_num);
        value.resize(v_num);
        is_active.resize(v_num);
        
        for(cnt_t j = 0; j < comm_bound_node[i].size(); ++j){ // 考虑当前聚类的每个边界点
            
            vid_t u = comm_bound_node[i][j]; // u为当前考虑的边界点

            std::fill(residue.begin(), residue.end(), 0.0);
            std::fill(value.begin(), value.end(), 0.0);
            std::fill(is_active.begin(), is_active.end(), false);
            residue[u] = 1.0; // send u a unit message
            active_v.push(u); // 将 u 加入 active 队列
            is_active[u] = true; //将 u 标记为已激活

            while(!active_v.empty()){ // 当没有节点激活后说明迭代计算结束
                vid_t a = active_v.front(); // 取激活队列的第一个节点 a
                active_v.pop();
                double r = residue[a];  
                residue[a] = 0.0; 
                is_active[a] = false; // 读取点 a 的 residue 后将a置为未激活，预防 a 有自环 
                value[a] += alpha * r; // 更新 a 的 value
                for(cnt_t k = 0; k < edge_0[a].size(); ++k){
                    vid_t v = edge_0[a][k];
                    if(vdata[v] != vdata[a])
                        continue;
                    residue[v] += r * (1 - alpha) / (double)edge_0[a].size(); // push 到 a 的邻点
                    if(is_bound[v] == false && is_active[v] == false){ // 当 a 的邻点不是边界点且该邻点还未被更新时考虑激活该点
                        if(residue[v] > thr){ // 激活邻点
                            active_v.push(v);
                            is_active[v] = true;
                        }
                    }
                }
            }
            
            //if(omp_get_thread_num() == 0)
            //    logfile << "the bound node " << j << " in low graph " << i << " run pagerank over" << std::endl;
            
            // 建立边界点之间的 short cuts
            for(cnt_t k = 0; k < comm_bound_node[i].size(); ++k){ // 遍历当前聚类的每个边界点
                vid_t v = comm_bound_node[i][k]; //考察 u 和 v 之间的 short cuts
                if(residue[v] > thr){
                    up_graph[u].push_back( std::make_pair(v, residue[v]) );
                }    
            }

            // 建立边界点到内部点的 assignment short cuts
            //vdata_t cur_comm = valid_comm_map[i];
            for(cnt_t k = 0; k < comm_inner_node[i].size(); ++k){
                vid_t v = comm_inner_node[i][k];
                if(value[v] > thr)
                    assign_graph[i][u].push_back( std::make_pair(v, value[v]) );
            }
        }
    }

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

    logfile << "output low graph" <<std::endl;

    std::cout << "comput short cuts end " << std::endl;
    
    // 输出下层图
    //std::ofstream outfile("/workspaces/GEN_LAYPH/output/test/low_graph.e");
    //std::ofstream outfile("/workspaces/GEN_LAYPH/output/LiveJournal/low_graph.e");
    std::ofstream outfile("/workspaces/GEN_LAYPH/output/uk2002/low_graph.e");
    //std::ofstream outfile("/workspaces/GEN_LAYPH/output/uk2005/low_graph.e");
    for(vdata_t i = 0 ; i < max_comm; ++i){
            cnt_t size_sum = 0;
            for(cnt_t j = 0; j < comm_inner_node[i].size(); ++j){
                vid_t u = comm_inner_node[i][j];
                size_sum += low_graph[i][u].size();
            }
            outfile << i << " " << size_sum << std::endl;
            for(cnt_t j = 0; j < comm_inner_node[i].size(); ++j){
                vid_t u = comm_inner_node[i][j];
                for(cnt_t k = 0; k < low_graph[i][u].size(); ++k){
                    vid_t v = low_graph[i][u][k];
                    outfile << u << " " << v << std::endl;
                }
            }
    }
    outfile.close();

    logfile << "output up graph" <<std::endl;

    // 输出上层图
    //outfile.open("/workspaces/GEN_LAYPH/output/test/up_graph.e");
    //outfile.open("/workspaces/GEN_LAYPH/output/LiveJournal/up_graph.e");
    outfile.open("/workspaces/GEN_LAYPH/output/uk2002/up_graph.e");
    //outfile.open("/workspaces/GEN_LAYPH/output/uk2005/up_graph.e");

    for(vid_t i = 0; i < v_num; ++i){
        std::sort(up_graph[i].begin(), up_graph[i].begin() + up_graph[i].size(), cmp);
        for(cnt_t j = 0; j < up_graph[i].size(); ++j){
            vid_t u = up_graph[i][j].first;
            weight_t w = up_graph[i][j].second;
            outfile << i << " " << u << " " << std::fixed << std::setprecision(10) <<  w << std::endl;
        }
    }
    outfile.close();

    logfile << "output assign graph" <<std::endl;
    
    //输出赋值图
    //outfile.open("/workspaces/GEN_LAYPH/output/test/assign_graph.e");
    //outfile.open("/workspaces/GEN_LAYPH/output/LiveJournal/assign_graph.e");
    outfile.open("/workspaces/GEN_LAYPH/output/uk2002/assign_graph.e");
    //outfile.open("/workspaces/GEN_LAYPH/output/uk2005/assign_graph.e");
    for(vdata_t i = 0 ; i < max_comm; ++i){
            cnt_t size = 0;
            for(cnt_t j = 0; j < comm_bound_node[i].size(); ++j){
                vid_t u = comm_bound_node[i][j];
                size += assign_graph[i][u].size();
            }
            outfile << i << " " << size << std::endl;
            for(cnt_t j = 0; j < comm_bound_node[i].size(); ++j){
                vid_t u = comm_bound_node[i][j];
                for(cnt_t k = 0; k < assign_graph[i][u].size(); ++k){
                    vid_t v = assign_graph[i][u][k].first;
                    weight_t w = assign_graph[i][u][k].second;
                    outfile << u << " " << v << " " << w << std::endl;
                }
            }
    }
    outfile.close();
  
    // 输出点所在图（ 上层图（本身就在上层图的点 + 下层图的边界点），下层图（内部点））
    //outfile.open("/workspaces/GEN_LAYPH/output/test/vertex_map.v");
    //outfile.open("/workspaces/GEN_LAYPH/output/LiveJournal/vertex_map.v");
    outfile.open("/workspaces/GEN_LAYPH/output/uk2002/vertex_map.v");
    //outfile.open("/workspaces/GEN_LAYPH/output/uk2005/vertex_map.v");
    for(vdata_t i = 0 ; i < v_num; ++i){
        if(vdata[i] != -1 && is_bound[i])
            outfile << i << " -1" << std::endl;
        else
            outfile << i << " " << vdata[i] << std::endl; 
        
    }
    outfile.close();

    //outfile.open("/workspaces/GEN_LAYPH/output/test/vertex_bound.v");
    // outfile.open("/workspaces/GEN_LAYPH/output/LiveJournal/vertex_bound.v");
    // for(vdata_t i = 0 ; i < max_comm; ++i){
    //     outfile << i << " " << comm_bound_node[i].size() << std::endl;
    //     for(cnt_t j = 0; j < comm_bound_node[i].size(); ++j){
    //         vid_t v = comm_bound_node[i][j];
    //         outfile << v << std::endl;
    //     }
    // }
    // outfile.close();

    logfile.close();

    std::chrono::duration<double> time_span = duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "It tooks " << time_span.count() << " ms." << std::endl;
    return 0;
}