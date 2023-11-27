import numpy as np
from collections import defaultdict


def evalSpice(filename):
    # Check if file is of correct type

    # Open file and parse it
    try:
        netlist = open(filename, "r")
    except:
        raise FileNotFoundError("Please give the name of a valid SPICE file as input")

    line = netlist.readline()

    # Parsing the txt file till .circuit
    def check(x, y):
        """
        Returns:
            True if string 'x' does not start with string 'y'
        """
        if len(x) < len(y):
            return True
        else:
            return x[0 : len(y)] != y

    # This reads the first few junk lines
    while check(line, ".circuit"):
        line = netlist.readline()
        if not line:
            raise ValueError("Malformed circuit file")

    # Now parsing the netlist
    Vsources = {}  # All voltage sources will be stored as:
    # Vsources[("ni","nj")]=value::float between node i and j, GND included
    # node i (the 1st specified node) is taken as +ve terminal and node j as -ve
    Vnames = []
    Branches = defaultdict(
        lambda: [0, 0]
    )  # All branch admittances and current sources will be stored as:
    # Branches{("ni","nj")}=[admitance::float,Isource::float], GND included
    # I is assumed to be from node j to i (2nd to 1st specified node)
    # Nodes connected by voltage sources  are not included in this dictionary
    Nodes = []
    # Nodes is a list that assigns an index to each node, including the "GND" node
    Nodes_index = {}
    # Nodes_index assigns an index to each node
    line = netlist.readline()
    while check(line, ".end"):
        linelist = line.split()
        if linelist[1] not in Nodes:
            Nodes.append(linelist[1])
        if linelist[2] not in Nodes:
            Nodes.append(linelist[2])
        if line[0] == "V":
            Vnames.append(linelist[0])
            if linelist[3] == "dc" or linelist[3] == "ac":
                # Check if dc or ac is specified
                try:
                    float(linelist[4])
                except:
                    raise ValueError("Malformed circuit file")
                try:
                    if linelist[5][0] != "#":
                        raise ValueError("Malformed circuit file")
                except:
                    pass
                if linelist[1] == linelist[2]:
                    raise ValueError("Circuit error: no solution")
                else:
                    Vsources[(linelist[1], linelist[2])] = float(linelist[4])
            else:
                try:
                    float(linelist[3])
                except:
                    raise ValueError("Malformed circuit file")
                try:
                    if linelist[4][0] != "#":
                        raise ValueError("Malformed circuit file")
                except:
                    pass
                if linelist[1] == linelist[2]:
                    raise ValueError("Circuit error: no solution")
                else:
                    Vsources[(linelist[1], linelist[2])] = float(linelist[3])
        elif line[0] == "I":
            if linelist[3] == "dc" or linelist[3] == "ac":
                try:
                    if linelist[5][0] != "#":
                        raise ValueError("Malformed circuit file")
                except:
                    pass
                try:
                    float(linelist[4])
                except:
                    raise ValueError("Malformed circuit file")
                # Check if dc or ac is specified
                if (linelist[1], linelist[2]) in Branches:
                    Branches[(linelist[1], linelist[2])][0] += float(linelist[4])
                    Branches[(linelist[2], linelist[1])][0] -= float(linelist[4])
                    # Adding I sources to get Is(eq) between 2 nodes
                elif linelist[1] == linelist[2]:
                    pass
                else:
                    Branches[(linelist[1], linelist[2])] = [float(linelist[4]), 0]
                    Branches[(linelist[2], linelist[1])] = [-float(linelist[4]), 0]
            else:
                try:
                    if linelist[4][0] != "#":
                        raise ValueError("Malformed circuit file")
                except:
                    pass
                try:
                    float(linelist[3])
                except:
                    raise ValueError("Malformed circuit file")
                if (linelist[1], linelist[2]) in Branches:
                    Branches[(linelist[1], linelist[2])][0] += float(linelist[3])
                    Branches[(linelist[2], linelist[1])][0] -= float(linelist[3])
                elif linelist[1] == linelist[2]:
                    pass
                else:
                    Branches[(linelist[1], linelist[2])][0] = [float(linelist[3]), 0]
                    Branches[(linelist[2], linelist[1])][0] = [-float(linelist[3]), 0]
        elif line[0] == "R":
            try:
                if linelist[4][0] != "#":
                    raise ValueError("Malformed circuit file")
            except:
                pass
            try:
                float(linelist[3])
            except:
                raise ValueError("Malformed circuit file")
            if (linelist[1], linelist[2]) in Branches:
                if float(linelist[3]) == 0:
                    Branches[(linelist[1], linelist[2])][1] += 2 ^ 50
                    Branches[(linelist[2], linelist[1])][1] += 2 ^ 50
                Branches[(linelist[1], linelist[2])][1] += 1 / float(linelist[3])
                Branches[(linelist[2], linelist[1])][1] += 1 / float(linelist[3])
                # Admittances are summed to get "Yeq" between 2 node
            elif linelist[1] == linelist[2]:
                pass
            else:
                if float(linelist[3]) == 0:
                    Branches[(linelist[1], linelist[2])][1] = [0, 2 ^ 50]
                    Branches[(linelist[2], linelist[1])][1] = [0, 2 ^ 50]
                Branches[(linelist[1], linelist[2])] = [0, 1 / float(linelist[3])]
                Branches[(linelist[2], linelist[1])] = [0, 1 / float(linelist[3])]
        else:
            raise ValueError("Only V, I, R elements are permitted")
        line = netlist.readline()
        if not line:
            raise ValueError("Malformed circuit file")

    # Now generate matrix representation and solve
    v = len(Vsources)
    n = len(Nodes)
    if n==0:
        raise ValueError("Circuit error: no solution")
    Y = np.zeros((n + v, n + v), dtype=np.float64)  # Admittance matrix
    B = np.zeros((n + v,), dtype=np.float64)  # Vector on RHS of linear system
    # All admittance added to Y and I sources to B in 2 loops
    for i in range(n):
        if Nodes[i] != "GND":
            for j in range(n):
                if i != j:
                    B[i] -= Branches[(Nodes[i], Nodes[j])][0]
                    Y[i, j] -= Branches[(Nodes[i], Nodes[j])][1]
        else:
            Y[i, i] += 1

    for i in range(n):
        if Nodes[i] != "GND":
            Y[i, i] -= np.sum(Y[i, 0:n])
    # Next all constraints due to voltage sources are added
    b = 0  # iterator
    Nodes_index = {Nodes[i]: i for i in range(n)}
    for node_pair in Vsources:
        Y[n + b, Nodes_index[node_pair[0]]] = 1.0
        Y[n + b, Nodes_index[node_pair[1]]] = -1.0
        B[n + b] = Vsources[node_pair]
        b += 1
    # Next all the unknown currents contribution to the nodal voltage equations are added
    b = 0  # iterator
    for node_pair in Vsources:
        if node_pair[0] != "GND" and node_pair[1] != "GND":
            Y[Nodes_index[node_pair[0]], n + b] += 1.0
            Y[Nodes_index[node_pair[1]], n + b] -= 1.0
        elif node_pair[0] == "GND":
            Y[Nodes_index[node_pair[1]], n + b] -= 1.0
        else:
            Y[Nodes_index[node_pair[0]], n + b] += 1.0

    # Solve the linear equations

    try:
        A = np.linalg.solve(Y, B)
    except:
        raise ValueError("Circuit error: no solution")

    # Creating the result dictionaries
    V = {Nodes[i]: A[i] for i in range(n)}
    I = {}
    for i in range(len(Vnames)):
        I[Vnames[i]] = A[n + i]

    return (V, I)



"""    
    print("\n V=",V,"\n\n")                        
    print("List of branch resistances and currents=",Branches,"\n")
    print("List of Voltage sources",Vsources)
    print("\n Admittance",Y,"\n")
    print("\nNode list",Nodes, Nodes_index,"\n")
    print("\n Rhs=",B,"\n")
"""
