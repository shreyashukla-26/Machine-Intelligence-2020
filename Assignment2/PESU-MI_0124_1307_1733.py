def node_remove(frontier,a):
	for i in range(0,len(frontier)):
		if(frontier[i][0] == a):
			return i

def DFS_Traversal(cost, start_point, goals):
	l = []
	frontier = []
	frontier.append([start_point,[start_point]])
	while(len(frontier)!=0):
		ele = frontier.pop(len(frontier)-1)
		l.append(ele)
		f = [f[0] for f in frontier]
		e = [e[0] for e in l]
		if(ele[0] in goals):
			return l[len(l)-1][1]
		for i in range(len(cost)-1,0,-1):
			if((i not in e) and (cost[ele[0]][i]>0)):
				if(i not in f):
					frontier.append([i,ele[1]+[i]])
				else:
					rem = node_remove(frontier, i)
					frontier.pop(rem)
					frontier.append([i,ele[1]+[i]])

	if(len(frontier)==0):
		return []
	
# UCS_Traversal
	
def UCS_Traversal(cost, start_point, goals):
    
	frontier = [[start_point, 7, []]]                           # Node = list(node, path_cost, path_to_node)     
                                                                # Node = list(int, int, list(int,...))
	explored = []
	while(len(frontier) != 0):

		frontier.sort(key = lambda x:x[2]+[x[0]])
		frontier.sort(key = lambda x:x[1])
		parent, p_cost, path = frontier.pop(0)                  # Pop the current node from the frontier
		explored.append([parent, p_cost, path])                 # Add node to the explored list
        
		if parent in goals:                                     # If the node is a goal state, return path to node + node
			return path + [parent]
        
		for i in range(1, len(cost)):                           # Explore the node
		        
			if (cost[parent][i] > 0):                           # If i is a child, create_node(i) -> node_number, path_cost, path_to_node
				child = i
				c_cost = p_cost + cost[parent][i]        
				c_path = path + [parent]
		            
				f = [f[0] for f in frontier]                    # Lists that hold the frontier nodes and explored nodes. (Only number)
				e = [e[0] for e in explored]
		            
				if((child not in f) and (child not in e)):      
					frontier.append([child, c_cost, c_path])
				else:                                           # else replace with lower path cost node
					for j in frontier:
						if(child == j[0]) and (j[1] > c_cost):
							j[1] = c_cost
							j[2] = c_path
							break
						elif(child == j[0]) and (j[1] == c_cost) and ((c_path + [child]) < (j[2] + [child])):
							j[1] = c_cost
							j[2] = c_path
							break   
        
	if(len(frontier) == 0):
		return []                                                             


# A-Star Traversal

def A_star_Traversal(cost, heuristic, start_point, goals):
    
	frontier = [[start_point, 0, []]]                           # Node = list(node, path_cost, path_to_node)     
                                                                # Node = list(int, int, list(int,...))
	explored = []
	while(len(frontier) != 0):
        
		frontier.sort(key = lambda x:x[2]+[x[0]])                      # Sort the frontier - (Priority Queue)
		frontier.sort(key = lambda x:x[1])
		parent, p_cost, path = frontier.pop(0)                  # Pop the current node from the frontier
		explored.append([parent, p_cost, path])                 # Add node to the explored list
        
		if parent in goals:                                     # If the node is a goal state, return path to node + node
			return path + [parent]
        
		for i in range(1, len(cost)):                           # Explore the node
            
			if (cost[parent][i] > 0):                           
				child = i
				c_cost = p_cost + cost[parent][i] + heuristic[child]      
				c_path = path + [parent] 
                
				f = [f[0] for f in frontier]                    # Lists that hold the frontier nodes and explored nodes. (Only number)
				e = [e[0] for e in explored]
                
				if((child not in f) and (child not in e)):      
					frontier.append([child, c_cost - heuristic[parent], c_path])  # Modified to subtract the heuristic of the parent 
				else:                                           				# else replace with lower path cost node
					for j in frontier:
						if(child == j[0]) and (j[1] > c_cost-heuristic[parent]):
							j[1] = c_cost-heuristic[parent]
							j[2] = c_path
							break
						elif(child == j[0]) and (j[1] == c_cost-heuristic[parent]) and ((c_path + [child]) < (j[2] + [child])):
							j[1] = c_cost-heuristic[parent]
							j[2] = c_path
							break   
        
	if(len(frontier) == 0):
		return []                                                         
   
# Tri Traversal
         
def tri_traversal(cost, heuristic, start_point, goals):

	l = []

	t1 = DFS_Traversal(cost, start_point, goals)
	t2 = UCS_Traversal(cost, start_point, goals)
	t3 = A_star_Traversal(cost, heuristic, start_point, goals)

	l.append(t1)
	l.append(t2)
	l.append(t3)
    
	return l    

