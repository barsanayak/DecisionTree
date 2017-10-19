


import pydot

def drawGraph():
    # this time, in graph_type we specify we want a DIrected GRAPH
    graph = pydot.Dot(graph_type='digraph')
    # creating nodes is as simple as creating edges!
    node_a = pydot.Node('Node A',label='Node A', style="filled", fillcolor="red")
    node_b = pydot.Node("Node B", style="filled", fillcolor="green")
    node_c = pydot.Node("Node C", style="filled", fillcolor="#0000ff")
    node_d = pydot.Node('Node D',label='Node A', style="filled", fillcolor="#976856")

    graph.add_node(node_a)
    graph.add_node(node_b)
    graph.add_node(node_c)
    graph.add_node(node_d)

    # finally we create the edges of keeping a reference to it in a variable
    graph.add_edge(pydot.Edge(node_a, node_b,label="and back we go again",labelfontcolor="#009933",fontsize="10.0", color="blue"))
    graph.add_edge(pydot.Edge(node_b, node_c))
    graph.add_edge(pydot.Edge(node_c, node_d))

    # and we are done
    graph.write_png('example8_graph.png')

def isPresent(graph,nodeName):
    node_a = pydot.Node(nodeName, style="filled", fillcolor="red")
    if(graph.get_node(node_a) is None):
        return True
    return False


dict1  = {'odor = a': 'e', 'odor = c': 'p', 'odor = f': 'p', 'odor = l': 'e', 'odor = m': 'p',
         'odor = n': {'spore-print-color = b': 'e', 'spore-print-color = h': 'e', 'spore-print-color = k': 'e',
                      'spore-print-color = n': 'e', 'spore-print-color = o': 'e', 'spore-print-color = r': 'p', 'spore-print-color = w':
                          {'habitat = d': {'gill-size = b': 'e', 'gill-size = n': 'p'}, 'habitat = g': 'e', 'habitat = l':
                              {'cap-color = c': 'e', 'cap-color = n': 'e', 'cap-color = w': 'p', 'cap-color = y': 'p'},
                           'habitat = p': 'e', 'habitat = w': 'e'}, 'spore-print-color = y': 'e'},
         'odor = p': 'p', 'odor = s': 'p', 'odor = y': 'p'}

dict2  = {'odor = a': 'e', 'odor = c': 'p', 'odor = f': 'p', 'odor = l': 'e', 'odor = m': 'p',
         'odor = n': {'spore-print-color = b': 'e', 'spore-print-color = h': 'e', 'spore-print-color = k': 'e'},
         'odor = p': 'p', 'odor = s': 'p', 'odor = y': 'p'}

dict3 = {'odor = a': 'e', 'odor = c': 'p', 'odor = f': 'p', 'odor = l': 'e', 'odor = m': 'p',
         'odor = n': 'p',
         'odor = p': 'p', 'odor = s': 'p', 'odor = y': 'p'}
def drawGraph2(graph,dictArg,str,nodesWithDict):
    if(dictArg == ""):
        return
    #graph.add_node(node_a)
    #graph.add_edge(pydot.Edge(node_a, node_b, label="and back we go again", labelfontcolor="#009933", fontsize="10.0",color="blue"))
    node_p =""
    val3 = ""
    if(len(nodesWithDict) != 0):
        print(nodesWithDict)
        node_p = nodesWithDict.pop()
    for k, v in dictArg.items():
        keyList = k.split('=')
        col = keyList[0].strip()
        val = keyList[1].strip()
        if(node_p == "" and isPresent(graph,col)):
            node_p = graph.get_node(col)
        elif(node_p == ""):
            node_p = pydot.Node(col, style="filled", fillcolor="red")
        if(type(v) is not dict):
            str = str + "a"
            node_c = pydot.Node(str, label=v, style="filled", fillcolor="red")
            graph.add_node(node_c)
            graph.add_edge(pydot.Edge(node_p, node_c, label=val, labelfontcolor="#009933", fontsize="10.0", color="blue"))
        else:
            str = str + "a"
            label2 = list(v.keys())[0]
            keyList2 = label2.split('=')
            col2 = keyList2[0].strip()
            val2 = keyList2[1].strip()
            val3 = v
            node_c2 = pydot.Node(str, label= col2, style="filled", fillcolor="red")
            graph.add_node(node_c2)
            graph.add_edge(pydot.Edge(node_p, node_c2, label=val, labelfontcolor="#009933", fontsize="10.0", color="blue"))
            nodesWithDict.append(node_c2)
    drawGraph2(graph, val3, str, nodesWithDict)


    return





graph = pydot.Dot(graph_type='graph')
str=""
nodesWithDict = []
drawGraph2(graph,dict1,str,nodesWithDict)
print(graph.get_nodes())
graph.write_png('example8_graph.png')