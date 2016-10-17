import tensorflow as tf

def show_graph(graph):
    print '===========\n graph description'
    for op in graph.get_operations(): 
        print op.name
    print '===========\n variables'
    #for v in graph.get_collection('variables'):
     #   print v.name
    #print '========'    

