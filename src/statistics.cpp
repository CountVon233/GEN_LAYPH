#include <vector>
#include <fstream>
#include <queue>
#include <iomanip>

using vid_t = int64_t;
using weight_t = double;
using vdata_t = int32_t;
using cnt_t = int64_t;

int main(){

    std::ifstream infile("/workspaces/gen_layph/dataset/LiveJournal-undirected.e");
    vid_t initial_cnt = 0;
    vid_t v;
    vid_t u;
    double w;
    while (infile >> u >> v) // vid and vd must start from 0
    {
        initial_cnt++;
    }
    infile.close();

    infile.open("/workspaces/gen_layph/output/up_graph.e");
    vid_t up_graph_cnt = 0;
    while (infile >> u >> v >> w) // edge from u to v (directed, no weight)
    {
        up_graph_cnt++;
    }
    infile.close();

    infile.open("/workspaces/gen_layph/output/low_graph.e");
    vid_t low_graph_cnt = 0;
    vdata_t gid;
    cnt_t e_num;
    while (infile >> gid >> e_num) // edge from u to v (directed, no weight)
    {
        for(cnt_t i = 0; i < e_num; ++i){
            infile >> u >> v;
            low_graph_cnt++;
        }
    }
    infile.close();

    std::ofstream outfile("/workspaces/gen_layph/output/statistics-output.txt");
    outfile << " number of edges in initial graph is " << initial_cnt << std::endl;
    outfile << " number of edges in obtained up graph is " << up_graph_cnt << std::endl;
    outfile << " number of edges in obtained low graph is " << low_graph_cnt << std::endl;
    outfile.close();
    return 0;
}