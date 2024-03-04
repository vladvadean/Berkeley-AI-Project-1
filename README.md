# PAC-MAN ASSIGNMENT #1
## Table of Contents
# Table of Contents
1. [The Purpose of the Project](#the-purpose-of-the-project)
2. [Depth First Search Algorithm](#depth-first-search-algorithm)
3. [Breadth First Search Algorithm](#breadth-first-search-algorithm)
4. [Uniform Cost Search Algorithm](#uniform-cost-search-algorithm)
5. [A* Search Algorithm](#a-search-algorithm)
6. [Finding All Corners Problem](#finding-all-corners-problem)
7. [Corner Problem Heuristic](#corner-problem-heuristic)
8. [Eating All Dots Problem](#eating-all-dots-problem)
9. [Reflex Agent](#reflex-agent)
10. [Minimax Agent](#minimax-agent)
11. [AlphaBeta Agent](#alphabeta-agent)
12. [Conclusion](#conclusion)

## The Purpose of the Project 
This project teaches us how to implement all types of multiple types of informed, uniformed, and adversarial search all being implemented in a Pac-Man game, for different mechanics such as the Pac-Man movement, the ghosts’ movement, and the decision-making of all characters. The project has many layouts that consist of 4 direction mazes implementing different mechanics depending on the part that is being tested such as: food mechanism, ghost mechanism, the power-ups etc. 
## Depth First Search Algorithm 
### Description 
The Depth First Search Algorithm is an uninformed type of search. It is being implemented for the Pac-Man character of the game and is being tested on a various number of layouts. The question is giving points depending on the number of scenarios that the Pac-Man reached the exit of the maze like layout of the game. The algorithm uses two stacks: one to stock the order of the nodes that are going to be processed and the other to stock the result of the search, being a list of cardinal directions. For the visited component of the nodes, the algorithm uses a set to store the coordinates of all the visited positions, because sets are more efficient at within access than lists. 
### Python Code 
```python
start_state = problem.getStartState() 
stack = util.Stack() 
visited = set() 
visited.add(start_state) 
start_node = [start_state, []] 
stack.push(start_node) 
while  not stack.isEmpty(): 
	current_state, moves = stack.pop() 
	visited.add(current_state)
	if problem.isGoalState(current_state): 
		return moves 
	for next_state in problem.getSuccessors(current_state): 
		if next_state[0] not  in visited: 
			new_moves = moves + [next_state[1]] 
			node_aux = [next_state[0], new_moves] 
			stack.push(node_aux)
```
### Observation 
The Depth First Search Algorithm is a riskier algorithm than Breadth First Search Algorithm. It will expand less nodes to find the solution in the first phase of the program, but if the goal is not found within the expanded nodes the algorithm will expand more nodes than Breadth First Search Algorithm, so it’s going to take more time to find the goal and less effective in this case. Unless we know in which cardinal order to expand the nodes and if it is consistent, for the average case Breadth First Search Algorithm would be a better choice.
## Breadth First Search Algorithm 
### Description 
Breadth First Search Algorithm is an uninformed type of search. It is being implemented for the Pac-Man character of the game and is being tested on a various number of layouts. The question is giving points depending on the number of scenarios that the Pac-Man reached the exit of the maze like layout of the game. The algorithm uses two queues: one to stock the order of the nodes that are going to be processed and the other to stock the result of the search, being a list of cardinal directions. For the visited component of the nodes, the algorithm uses a set to store the coordinates of all the visited positions, because sets are more efficient at within access than lists. 
### Python Code 
```python 
start_state = problem.getStartState() 
queue = util.Queue() 
visited = set() 
visited.add(start_state) 
start_node = [start_state, []] 
queue.push(start_node) 
while  not queue.isEmpty(): 
	current_state, moves = queue.pop() 
	if problem.isGoalState(current_state): 
		return moves 
	for next_state in problem.getSuccessors(current_state): 
		if next_state[0] not  in visited: 
			new_moves = moves + [next_state[1]] 
			node_aux = [next_state[0], new_moves] visited.add(next_state[0]) 
			queue.push(node_aux)
```
### Observation 
The Breadth First Search Algorithm is a safer way to search the goal state in such situations. Comparing both DFS’s and BFS’s worst cases the BFS’s expands less nodes. But in the best case of both algorithm DFS expands less nodes, but the chance of it happening is very low. Unless we have a consistent order in which nodes can be expanded and it is known the Breadth First Search Algorithm is a more reliable choice. 
## Uniform Cost Search Algorithm 
### Description
Uniform Cost Search Algorithm is an uninformed type of search. What differs from the other two types of informed search is that we consider bonus information about the nodes: the weight of an edge or the cost to get from one location to another. In this case we use a priority queue instead of a stack or a queue. We sort the nodes that are going to be extended by the total cost from the initial location to the current location. Every element of the priority queue will contain: the coordinates of that location, the list of moves required from the start node to get to the current node and the total cost needed to get here. We will still use a set for the visited attribute of the nodes. 
### Python Code 
```python
start_state = problem.getStartState()
priority_queue = util.PriorityQueue()
visited = set()
# (x, y), action, cost
start_node = (start_state, [], 0)
priority_queue.push(start_node, 0)

while not priority_queue.isEmpty():
    current_state, moves, cost = priority_queue.pop()
    if current_state not in visited:
        visited.add(current_state)
        if problem.isGoalState(current_state):
            return moves
        for next_state in problem.getSuccessors(current_state):
            new_moves = moves + [next_state[1]]
            node_aux = (next_state[0], new_moves, next_state[2] + cost)
            priority_queue.push(node_aux, next_state[2] + cost)

``` 
### Observation 
If the layout contains a diverse set of costs from one node to another. In this case the shortest path will not be influenced only by the number of nodes of the path but of the total cost too. The algorithm should consider the weight attribute as the most important information about the node. For this kind of problem the Uniform Cost Search should be the algorithm used to find he goal state.
## A* Search Algorithm 
### Description 
The A* Search Algorithm is a informed search, meaning we know where the goal state is. This case of problems provide bonus information: g(n) = the actual cost from the start node to the current node, h(n) = the estimated cost of the best path that continues from n to a goal (heuristic). The heuristic shall be implemented in the next questions. For UCS the cost is f(n)=g(n), making decisions considering only the cost of the expanded nodes. 
### Python Code 
```python
priority_queue = util.PriorityQueue()
# (x, y), cost
visited = set()
start_state = problem.getStartState()
start_node = (start_state, [], 0)
priority_queue.push(start_node, 0)

while not priority_queue.isEmpty():
    current_state, moves, cost = priority_queue.pop()
    visited.add((current_state, cost))
    if problem.isGoalState(current_state):
        return moves
    for next_node in problem.getSuccessors(current_state):
        new_moves = moves + [next_node[1]]
        new_cost = problem.getCostOfActions(new_moves)
        node_aux = (next_node[0], new_moves, new_cost)
        in_visited = False
        for visited_node in visited:
            if (next_node[0] == visited_node[0]) and (new_cost >= visited_node[1]):
                in_visited = True
                break
        if not in_visited:
            priority_queue.push(node_aux, new_cost + heuristic(next_node[0], problem))
            visited.add((next_node[0], new_cost))
```
### Observation
 For the A* Search Algorithm to be efficient the heuristic must be a consistent one, meaning it should order the nodes in the real order, considering the total cost, from the goal state. The better the heuristic the less nodes are expanded and the more efficient the algorithm is. 
## Finding All Corners Problem 
### Description 
 This problem’s goal is to find the shortest path for the Pac-Man to reach all four corners of the maze depending on the layout. For this problem we must enhance the start state with additional information about which corners were already reached. To solve this problem I implemented a tuple containing the coordinates of the corners that were already reached. For checking if the actual state is a goal state the length of the tuple must be 4. 
### Python Code
 ```python 
 def getStartState(self):
    """
    Returns the start state (in your state space, not the full Pacman state space).
    """
    # start_state = (starting position, checked corners)
    start = self.startingPosition
    corners = tuple()
    start_state = (start, corners)
    return start_state

def isGoalState(self, state: Any):
    """
    Returns whether this search state is a goal state of the problem.
    """
    # Append the coordinates (x, y) to the visited corners
    visited_corners = state[1]
    a = list(visited_corners)
    if state[0] in self.corners and state[0] not in a:
        a.append(state[0])
    visited_corners = tuple(a)
    if len(visited_corners) == 4:
        return True
    else:
        return False

def getSuccessors(self, state: Any):
    """
    Returns successor states, the actions they require, and a cost of 1.
    As noted in search.py:
    For a given state, this should return a list of triples, (successor, action, stepCost),
    where 'successor' is a successor to the current state, 'action' is the action required to get there,
    and 'stepCost' is the incremental cost of expanding to that successor.
    """
    successors = []
    currentPosition, visited_corners = state
    for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
        # Add a successor state to the successor list if the action is legal
        # Here’s a code snippet for figuring out whether a new position hits a wall:
        x, y = currentPosition
        dx, dy = Actions.directionToVector(action)
        nextx, nexty = int(x + dx), int(y + dy)
        # Get the layout of the map: 1-wall, 0-free
        wall_hit = self.walls[nextx][nexty]
        if not wall_hit:
            aux_corners = list(visited_corners)
            new_node = (nextx, nexty)
            if new_node in self.corners and new_node not in aux_corners:
                aux_corners.append(new_node)
            visit_corners = tuple(aux_corners)
            successor = ((new_node, visit_corners), action, 1)
            successors.append(successor)

    self.expanded += 1  # DO NOT CHANGE
    return successors
```
### Observation 
The hardest bug to fix was to implement an additional structure. The program requests the structure to be hash-able so we need to transmit for example by a tuple and convert it to a list when it requires the elements to be manipulated.
## Corner Problem Heuristic 
### Description 
This questions goal is to implement a consistent heuristic that helps to expand as few nodes as possible for the problem mentioned above. 
###Python Code 
```python 
# Sum of the unvisited corners does not work for the test cases, it expands fewer nodes but takes a longer path for the case
# python3 pacman.py -l mediumCorners -p AStarCornersAgent -z 0.5
# Also tried with the sum//no_corners not visited but for the last test case it expands too many nodes

sum_distance = 0
max_distance = 0
no_corners = 0

for corner in corners:
    if corner not in state[1]:  # if the corner has not been visited
        no_corners += 1
        sum_distance += mazeDistance(state[0], corner, problem.gameState)
        current_distance = mazeDistance(state[0], corner, problem.gameState)
        if max_distance < current_distance:
            max_distance = current_distance

if no_corners == 0:
    return 0
else:
    return sum_distance // no_corners  # Average distance to unvisited corners
```
### Observation 
The best heuristic I found is the longest distance from Pac-Man to a not visited corner that checks all requirements. I tried to implement an heuristic that computes 16 the sum of the Pac-Man to all the not visited corners but the program fails in this case because the heuristic guesses a longer path than the real path. This heuristic didn’t work for the automated check test cases, but for the manual test cases it compiled and even expanded less nodes than the heuristic implemented above. So having a too large value i tried to get the average distance to all not visited corners by diving the sum from before with the number of not visited nodes. 
## Eating All Dots Problem 
### Description
 This heuristic is part of the AStarFoodSearchAgent and should help track all the food remaining on the layout of the maze. 
### Python Code
```python
# Find the furthest food from the pacman location
# Get the coordinates from foodGrid
start_position = problem.startingGameState
food_positions = foodGrid.asList()
max_distance = 0
if len(food_positions) == 0:
    return 0
for food in food_positions:
    dist = mazeDistance(start_position, food, problem.gameState)
    if dist > max_distance:
        max_distance = dist
return max_distance

```
### Observation 
This heuristic is similar to the one above only difference is that there are more locations with food on the layout of the map. In this case the heuristic implemented is the the furthest food from the Pac-Man location.
## Reflex Agent 
### Description 
The Reflex Agent is part of the adversarial search. This component helps the Pac-Man make the most efficient choices by evaluating the possible moves it can make. The Reflex Agent problem requires the user to define a better evaluation function to move as few position as possible, but the most important requirement is to not be caught by the ghosts. 
### Python Code
```python 
succe s so rGameS ta te = currentGameState . gene ra tePacmanSucce s so r ( a c ti o n ) newPos = succe s so rGameS ta te . ge tPacmanPo si tion ( ) newGho s tPo si tion s = succe s so rGameS ta te . g e tG h o s tP o si ti o n s ( ) ” ∗∗∗ YOUR CODE HERE ∗∗∗ ” # a b i g g e r w e i g h t f o r t h e t o t a l d i s t a n c e away from g h o s t s emerges v i c t o r i o u s b u t a t a v e ry low s c o r e # curren tGameS ta te . g e t S c o r e ( ) + ghos tD is tanceComponen t ∗ 1000 + foodD is tanceComponen t # ghos tD is tanceComponen t = ( nex tTo talD is tanceAwayFromGhos ts − curren tTo talD is tanceAwayFromGhos ts ) # foodD is tanceComponen t = ( curren tTo talD is tanceAwayFromFood − nextTo talD istanceAwayFromFood ) # t h e more complex f o rmul a t h e worse foodComponent = 9999999 minGhostDistance = 999999999 for p o s i t i o n in newGho s tPo si tion s : d i s t = manhattanDistance ( p o si ti o n , newPos ) i f minGhostDistance < d i s t : minGhostDistance = d i s t 19 i f minGhostDistance == 999999999: minGhostDistance = 1 for food in succe s so rGameS ta te . getFood ( ) . a s L i s t ( ) : foodComponent = min( foodComponent , manhattanDistance ( newPos , food ) ) for gho s t in newGho s tPo si tion s : i f ( manhattanDistance ( newPos , gho s t ) < 2 ) : return −999999 return succe s so rGameS ta te . g e t S c o r e ( ) + 1. 0 / minGhostDistance + 1. 0 / foodComponent 
```
### Observation
 I tried to take into account the current state of the Pac-Man and the next state by difference of the total distance away from food and the total distance away from ghosts. All of these elements were of the same importance. The result didn’t even win in all cases. So i tried to only take into account the distances of the next state. It was better but not getting 4 points, because the paths were not efficient. So I evaluated only the minimum distance of the food and of the ghost and automatically return a very bad score if a ghost is right next to you and it works. The weight of the component must differ since they are not all of the same importance. 
## Minimax Agent 
### Description 
This algorithm is used to compute many more moves ahead of the current state taking into account the possible successor states of all ghosts. The structure of the game is looking alike a tree toggling from one level to another in which either the Pac-Man chooses the max amount of all the possible scores, either the ghosts chooses the min amount.
### Python Code
```python
# agentIndex == 0 for pacman
# agentIndex != 0 for ghosts
def processingMinMax(gameState, depth, agentIndex):
    if agentIndex == 0:
        value = -999999
    else:
        value = 999999

    if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)

    if agentIndex == 0 and depth + 1 == self.depth:
        return self.evaluationFunction(gameState)

    legalActions = gameState.getLegalActions(agentIndex)
    for action in legalActions:
        successorState = gameState.generateSuccessor(agentIndex, action)
        if agentIndex == 0:
            value = max(value, processingMinMax(successorState, depth + 1, 1))
        else:
            if agentIndex == (gameState.getNumAgents() - 1):
                value = min(value, processingMinMax(successorState, depth, 0))
            else:
                value = min(value, processingMinMax(successorState, depth, agentIndex + 1))
    return value
# Start to generate the decision tree starting with the pacman
actions = gameState.getLegalActions(0)
currentScore = -999999
result = ''
for action in actions:
    nextState = gameState.generateSuccessor(0, action)
    score = processingMinMax(nextState, 0, 1)
    if score > currentScore:
        result = action
        currentScore = score
return result

 ``` 
### Observation 
The algorithm uses the same method depending on the index of the agent that is being processed for. The project has enumerated the agents in such manner that 0 is always reserved for the Pac-Man and all the others are 22 reserved for the remaining number of ghosts that differs from a layout or a test case to another. But first before calling the method we must first process the state of the Pac-Man and after that expand the whole decision tree. 
## AlphaBeta Agent 
### Description 
This algorithm is an enhanced version of the Minimax Algorithm. The AlphaBeta algorithm does not expand all the children trees if the values do not overwrite the values that are already checking the requirements, being a more efficient algorithm that the one before, speed and memory wise. 
### Python Code 
```python
# agentIndex == 0 for pacman
# agentIndex != 0 for ghosts
def processingMinMax(gameState, depth, agentIndex, alpha, beta):
    if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)
    
    if agentIndex == 0 and depth + 1 == self.depth:
        return self.evaluationFunction(gameState)
    
    if agentIndex == 0:
        value = -9999999
    else:
        value = 9999999
    
    legalActions = gameState.getLegalActions(agentIndex)
    alphaCopy = alpha
    betaCopy = beta
    
    for action in legalActions:
        successor = gameState.generateSuccessor(agentIndex, action)
        if agentIndex == 0:
            value = max(value, processingMinMax(successor, depth + 1, 1, alphaCopy, beta))
            if value > beta:
                return value
            alphaCopy = max(alphaCopy, value)
        else:
            if agentIndex == (gameState.getNumAgents() - 1):
                value = min(value, processingMinMax(successor, depth, 0, alpha, betaCopy))
            else:
                value = min(value, processingMinMax(successor, depth, agentIndex + 1, alpha, betaCopy))
            if value < alpha:
                return value
            betaCopy = min(betaCopy, value)
    return value

# Start by the root processing the pacman
actions = gameState.getLegalActions(0)
currentScore = -999999
returnAction = ''
alpha = -999999
beta = 999999

for action in actions:
    nextState = gameState.generateSuccessor(0, action)
    score = processingMinMax(nextState, 0, 1, alpha, beta)
    if score > currentScore:
        returnAction = action
        currentScore = score
    if score > beta:
        return returnAction
    alpha = max(alpha, score)

return returnAction

```
### Observation 
The algorithm is based on the same as methods and structure as Minimax algorithm and the same principle of agents indexing. The modification brought to the previous version was the bonus logic required to compare the alpha and the beta with the new values computed and to take action based on the new modifications of the values. 
## Conclusion
 This project goals are to learn more topics about the fundamentals of Artificial Intelligence through a more fun and an easier way to visualize all the modification brought to the code and to test it. Unfortunately all the documentation files are not enough to know everything you need about how to code and fulfill the requirements. There are so many scripts and classes that are used and not mentioned and so many mechanics used and not mentioned. For example I didn’t know there are capsules which can be eaten and scare the ghosts and eat them for extra points, making the Pac-Man invincible. Another example is that there are implemented methods for the class gameState such as isWin() or isLose() but there were mentioned nowhere. The structures which are frequently used such as start state are not described in any part of the documentation so you must print them and do a continuous loop of try and error, instead of a document explaining it. You should first check all the classes, the methods and the logic created by the evaluator first then try to code, and that is the documentations purpose.
