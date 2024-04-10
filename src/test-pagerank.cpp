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

using vid_t = int64_t;
using weight_t = double;
using vdata_t = int32_t;
using cnt_t = int64_t;

vid_t v_num = 0;
vdata_t max_comm = 0;
double alpha = 0.15;
double thr = 1e-10;
 

std::vector <vdata_t> vdata;
std::vector< std::vector<vid_t> > comm_bound_node;
std::vector<std::map<vid_t, std::vector<vid_t>> > low_graph;
std::vector<std::vector<vid_t> > low_graph_vertex;
std::vector< std::vector < std::pair<vid_t, weight_t> > > up_graph;
std::vector<vid_t> up_graph_vertex;
std::vector< std::map< vid_t, std::vector< std::pair<vid_t, weight_t> > > >  assign_graph;

std::vector<cnt_t> degree_low_graph, degree_0;
std::vector < std::vector<vid_t> > edge_0;

void Init(){
    v_num = vdata.size();

    comm_bound_node.clear();
    comm_bound_node.resize(max_comm);
    
    low_graph.resize(max_comm);
    assign_graph.resize(max_comm);
    for(vdata_t i = 0; i < max_comm; ++i){
        low_graph[i].clear();
        assign_graph[i].clear();
    }
    up_graph.resize(v_num);

    low_graph_vertex.clear();
    low_graph_vertex.resize(max_comm);
    up_graph_vertex.clear();

    edge_0.clear();
    edge_0.resize(v_num);

    degree_0.clear();
    degree_0.resize(v_num);
    degree_low_graph.clear();
    degree_low_graph.resize(v_num);
}

int main(){

    std::cout << 1 << std::endl;

    std::ifstream infile("/workspaces/gen_layph/output/test/vertex_map.v");
    //std::ifstream infile("/workspaces/gen_layph/output/LiveJournal/vertex_map.v");
    vid_t v, u;
    vdata_t vd;
    while (infile >> v >> vd) // vid and vd must start from 0
    {
        vdata.push_back(vd); //确定每个点及其所在聚类
        max_comm = std::max(max_comm, vd); //得到聚类范围
    }
    max_comm++;
    infile.close();

    std::cout << 2 << std::endl;

    Init();

    std::cout << 3 << std::endl;

    for(vid_t i = 0; i < v_num; ++i){
        if(vdata[i] == -1)
            up_graph_vertex.push_back(i);
        else
            low_graph_vertex[vdata[i]].push_back(i);
    }

    std::cout << 4 << std::endl;

    //infile.open("/workspaces/gen_layph/output/test/vertex_bound.v");
    infile.open("/workspaces/gen_layph/output/LiveJournal/vertex_bound.v");
    vdata_t comm_id;
    cnt_t comm_size;
    while (infile >> comm_id >> comm_size) // comm id and number of bound nodes
    {
        for(cnt_t i = 0; i < comm_size; ++i){
            infile >> u;  // edge from u to v (directed, no weight)
            comm_bound_node[comm_id].push_back(u);
            low_graph_vertex[comm_id].push_back(u);
        }
    }
    infile.close();

    std::cout << 5 << std::endl;

    //infile.open("/workspaces/gen_layph/output/test/low_graph.e");
    infile.open("/workspaces/gen_layph/output/LiveJournal/low_graph.e");
    while (infile >> comm_id >> comm_size) // comm id and number of edges
    {
        for(cnt_t i = 0; i < comm_size; ++i){
            infile >> u >> v;  // edge from u to v (directed, no weight)
            degree_low_graph[u]++;
            low_graph[comm_id][v].push_back(u);
        }
    }
    infile.close();

    std::cout << 6 << std::endl;

    //infile.open("/workspaces/gen_layph/output/test/up_graph.e");
    infile.open("/workspaces/gen_layph/output/LiveJournal/up_graph.e");
    weight_t w;
    while (infile >> u >> v >> w) // edge from u to v with weight
    {
        up_graph[v].push_back( std::make_pair(u, w));
    }
    infile.close();

    std::cout << 7 << std::endl;

    //infile.open("/workspaces/gen_layph/output/test/assign_graph.e");
    infile.open("/workspaces/gen_layph/output/LiveJournal/assign_graph.e");
    while (infile >> comm_id >> comm_size) // comm id and number of edges
    {
        for(cnt_t i = 0; i < comm_size; ++i){
            infile >> u >> v >> w;  // edge from u to v (directed, no weight)
            assign_graph[comm_id][u].push_back(std::make_pair(v, w));
        }
    }
    infile.close();

    std::cout << 8 << std::endl;

    vid_t source = 0;

    std::vector<double> residue(v_num, 0.0);
    std::vector<double> next_residue(v_num, 0.0);
    std::vector<double> value(v_num, 0.0); 
    std::vector<double> value_1(v_num, 0.0);

    //std:: ofstream outfile("/workspaces/gen_layph/output/test/standard_PPR");   
    std:: ofstream outfile("/workspaces/gen_layph/output/LiveJournal/standard_PPR_source=0"); 
    
    std::fill(residue.begin(), residue.end(), 0.0);
    std::fill(next_residue.begin(), next_residue.end(), 0.0);
    std::fill(value.begin(), value.end(), 0.0);

    residue[source] = 1.0;
    int active_cnt = 0;

    if(vdata[source] != -1){
        vdata_t cur_comm = vdata[source];
        while(1){
            active_cnt = 0;
            #pragma omp parallel for
            for(cnt_t i = 0; i < low_graph_vertex[cur_comm].size(); ++i){
                vid_t u = low_graph_vertex[cur_comm][i];
                for(cnt_t j = 0; j < low_graph[cur_comm][u].size(); ++j){
                    vid_t v = low_graph[cur_comm][u][j];
                    next_residue[u] += residue[v] * (1 - alpha) / (double)degree_low_graph[v];
                }
            }

            #pragma omp parallel for reduction(+ : active_cnt)
            for(cnt_t i = 0; i < low_graph_vertex[cur_comm].size(); ++i){
                vid_t u = low_graph_vertex[cur_comm][i];
                if(next_residue[u] > thr && vdata[u] != -1){
                    active_cnt = active_cnt + 1;
                }
                if(vdata[u] != -1){
                    value[u] += residue[u] * alpha;
                    residue[u] = 0.0;
                }
                residue[u] += next_residue[u];
                next_residue[u] = 0.0;
            }

            if(active_cnt == 0)
                break;
        } 
    }

    std::cout << 9 << std::endl;

    while(1){
        active_cnt = 0;
        #pragma omp parallel for
        for(cnt_t i = 0; i < up_graph_vertex.size(); ++i){
            vid_t u = up_graph_vertex[i];
            for(cnt_t j = 0; j < up_graph[u].size(); ++j){
                vid_t v = up_graph[u][j].first;
                weight_t w = up_graph[u][j].second;
                next_residue[u] += residue[v] * w;
            }
        }

        #pragma omp parallel for reduction(+ : active_cnt)
        for(cnt_t i = 0; i < up_graph_vertex.size(); ++i){
            vid_t u = up_graph_vertex[i];
            if(next_residue[u] > thr){
                active_cnt = active_cnt + 1;
            }
            value[u] += residue[u] * alpha;
            residue[u] = next_residue[u];
            next_residue[u] = 0.0;
        }

        if(active_cnt == 0)
            break;
    } 

    for(vdata_t i = 0; i < max_comm; ++i){
        for(cnt_t j = 0; j < comm_bound_node[i].size(); ++j){
            vid_t u = comm_bound_node[i][j];
            for(cnt_t k = 0; k < assign_graph[i][u].size(); ++k){
                vid_t v = assign_graph[i][u][k].first;
                weight_t w = assign_graph[i][u][k].second;
                value[v] += value[u] / alpha * w;
            }
        }
    }


    for(vid_t i = 0; i < v_num; ++i){
        outfile << i << " " << std::fixed << std::setprecision(15) << value[i] << std::endl;
    }
    
    outfile.close();

    return 0;
}