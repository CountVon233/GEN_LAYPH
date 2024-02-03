生成双层图结构
1、make run_filter_louvain (src/pre-filter-comm.cpp)
    2 input: (1) louvain_vertex_result: 原始图数据运行louvain的输出结果，为节点id和节点所在聚类号
            (2) edge_file: 原始图数据
    1 output: two_layer_vertex_map，为genlayph.cpp输入
2、make run_genlayph (src/genlayph.cpp)
    2 input: (1) two_layer_vertex_map: 对louvain输出结果进行预处理后的节点所在聚类号
            (2) edge_file: 原始图数据
    4 output: low_graph.e, up_graph.e, assign_graph.e, vertex_map.e

需要在源文件中修改输入输出文件路径# GEN_LAYPH
