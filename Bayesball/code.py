def create_graph():
    """Reads graph.txt and returns a dictionary
    with nodes as keys and the value is a list of
    nodes that the given node has a directed edge to.

    Returns:
        dict: the graph as a dictionary
    """
    with open('graph.txt', 'r') as g_file:
        K = int(g_file.readline())
        graph = {i: [] for i in range(1, K + 1)}
        for line in g_file:
            i, j = map(int, line.split())
            graph[i].append(j)
    return graph


def read_queries():
    """Reads queries.txt and returns a list of X, Y, Z
    triplets.

    Returns:
        list: the list of queries
    """
    with open('queries.txt', 'r') as q_file:
        queries = []
        for line in q_file:
            X, Y, Z = [], [], []
            x, y, z = line.split()
            X.extend(map(int, filter(bool, x[1:-1].split(','))))
            Y.extend(map(int, filter(bool, y[1:-1].split(','))))
            Z.extend(map(int, filter(bool, z[1:-1].split(','))))
            queries.append([X, Y, Z])
    return queries


def is_independent(graph, X, Y, Z):
    """Checks if X is conditionally indepedent
    of Y given Z.

    Args:
        graph (dict): the Bayesian network
        X (list): list of nodes in set X
        Y (list): list of nodes in set Y
        Z (list): list of nodes in set Z

    Returns:
        bool: True if X is conditionally indepedent
    of Y given Z, False otherwise.
    """
    # TODO
    # Phase 1 : Find all the unbloked v-structures
    L = Z.copy();
    A = [];
    while len(L) != 0:
        c = L.pop();
        if c not in set(A):
            parents_c = [nodes for (nodes, edge_to) in graph.items() if c in set(edge_to)]
            for parent in parents_c:
                L.append(parent)
            A = A + [c]

    # Phase 2 : Traverse all the trails starting from X
    T = X
    nodes_not_visited = list(graph.keys())
    i = 0
    while len(T) > 0 or i < 100 or len(nodes_not_visited) > 0:
        i = i + 1
        N = T.pop(0)
        nodes_not_visited.remove(N)

        # If N is not looked
        if N not in Z:
            for descendent in graph[N]:
                if descendent in Y:
                    return False
                if descendent in nodes_not_visited:
                    T.append(descendent)

        # If N is in A
        if N in A:
            for descendent in graph[N]:
                if descendent in Z:
                    parents_v_structure = [nodes for (nodes, edge_to) in graph.items() if (descendent in set(edge_to) and nodes is not N)]
                    for parent in parents_v_structure:
                        T.append(parent)
                        if parent in Y:
                            return False
        if len(T) == 0:
            return True


    return True


if __name__ == '__main__':
    graph = create_graph()
    Qs = read_queries()
    for X, Y, Z in Qs:
        output = 1 if is_independent(graph, X, Y, Z) else 0
        print(output)