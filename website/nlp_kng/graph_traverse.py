import networkx as nx
from . import config
from . import synonyms_extractor


@config.timer
def graph_traverse(nxGraph,svo_edge_labels,sourceNode, targetNode, verb, svodf):
    sourceNodes = []
    targetNodes = []
    noSrcTrgNode = False
    # print(f'all Nodes --> {nxGraph.nodes}')
    # print(f'svo_edge_labels --> {svo_edge_labels}')
    for node in nxGraph.nodes:
        # print('nodes -->', node, sourceNode, targetNode)
        if str(sourceNode).lower() in str(node).lower() or str(node).lower() in str(sourceNode).lower():
            sourceNodes.append(node)
        if str(targetNode).lower() in str(node).lower() or str(node).lower() in str(targetNode).lower() :
            targetNodes.append(node)
    all_paths = []
    # print(f'sourceNodes {sourceNode}--> {sourceNodes}')
    # print(f'targetNodes {targetNode}--> {targetNodes}')
    if len(sourceNodes) > 0 and len(targetNodes) > 0:
        for srcNode in sourceNodes:
            for trgNode in targetNodes:
                all_paths.append(list(nx.all_simple_paths(nxGraph, source = srcNode, target = trgNode)))
                all_paths.append(list(nx.all_simple_paths(nxGraph, source = trgNode, target = srcNode)))
        # print(f'all_paths --> {all_paths}')
        # all_paths = list(set(all_paths))
    else:
        if len(sourceNodes) == 0:
            if len(targetNodes) > 0:
                for trgNode in targetNodes:
                    for node in nxGraph.nodes:
                        all_paths.append(list(nx.all_simple_paths(nxGraph, source = trgNode, target = node)))
                        noSrcTrgNode = True
        if len(targetNodes) == 0:
            if len(sourceNodes) > 0:
                for srcNode in sourceNodes:
                    for node in nxGraph.nodes:
                        all_paths.append(list(nx.all_simple_paths(nxGraph, source = srcNode, target = node)))
                        noSrcTrgNode = True
    if noSrcTrgNode:
        # print(svodf)
        # print(verb)
        verb_synonyms = synonyms_extractor.get_synonyms([verb,'is'],svodf)
    if len(all_paths) > 0:
        full_paths = []
        for paths in all_paths:
            if len(paths) > 0:
                for path in paths:
                    # print(f'Path --> {path}')
                    path_len = len(path)
                    i = 0
                    full_path = ''
                    full_path_list = []
                    while i < path_len-1:
                        source = path[i]
                        target = path[i+1]
                        # print(f'noSrcTrgNode --> {source} {target} {noSrcTrgNode}')
                        if noSrcTrgNode == False:
                            relation = svo_edge_labels[tuple((source,target))]
                            # full_path = full_path +''+source + '-' + relation + '-' + target + '. '
                            relPath = source + '-' + relation + '-' + target + '. '
                            if relPath in full_path_list:
                                pass
                            else:
                                full_path_list.append(source + '-' + relation + '-' + target + '. ')
                        else:
                            relation = svo_edge_labels[tuple((source,target))]
                            # print(f'else relation {relation}')
                            # print(f'verb_synonyms --> {verb_synonyms}')
                            if len(verb_synonyms) > 0:
                                for verb_syn in verb_synonyms:
                                    if relation in verb_syn:
                                        # full_path = full_path +''+source + '-' + relation + '-' + target + '. '
                                        relPath = source + '-' + relation + '-' + target + '. '
                                        if relPath in full_path_list:
                                            pass
                                        else:
                                            full_path_list.append(source + '-' + relation + '-' + target + '. ')
                        i += 1
                    if len(full_path_list) > 0:
                        # full_path_list = list(set(full_path_list))
                        # print(full_path_list)
                        for fpl in full_path_list:
                            full_path = full_path + fpl
                    if len(full_path) > 0:
                        full_paths.append(full_path.strip())
        fps = []
        for fp in full_paths:
            for p in fp.split('.'):
                fps.append(p)
        # print(f'fps --> {fps}')
        nfps = []
        [nfps.append(x.strip()) for x in fps if x.strip() not in nfps and len(x) > 0]
        # print(f'nfps --> {nfps}')
        if len(nfps) > 0:
            return nfps
        # if len(full_paths) > 0:
            # print(f'Full Paths --> {full_paths}')
            # return list(set(full_paths))
    return []

