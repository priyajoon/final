#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace chrono;

class Graph {
private:
    int V;
    vector<vector<int>> adj;

public:
    Graph(int vertices) {
        V = vertices;
        adj.resize(V);
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // ✅ Recursive Sequential DFS
    void sequentialDFSUtil(int node, vector<bool> &visited) {
        visited[node] = true;
        cout << node << " ";

        for (int neighbor : adj[node]) {
            if (!visited[neighbor]) {
                sequentialDFSUtil(neighbor, visited);
            }
        }
    }

    void sequentialDFS(int start) {
        vector<bool> visited(V, false);
        cout << "Sequential DFS Traversal starting from node " << start << ": ";
        sequentialDFSUtil(start, visited);
        cout << endl;
    }

    // ✅ Recursive Parallel DFS
    void DFS_recursive(int node, vector<bool> &visited) {
        // Mark the current node as visited
        #pragma omp critical
        {
            if (!visited[node]) {
                visited[node] = true;
                cout << node << " ";
            } else {
                return; // Already visited
            }
        }

        // Traverse neighbors in parallel using tasks
        for (int neighbor : adj[node]) {
            #pragma omp task firstprivate(neighbor)
            {
                bool shouldVisit = false;
                #pragma omp critical
                {
                    shouldVisit = !visited[neighbor];
                }
                if (shouldVisit)
                    DFS_recursive(neighbor, visited);
            }
        }
    }

    void parallelDFS(int start) {
        vector<bool> visited(V, false);

        cout << "Parallel DFS Traversal starting from node " << start << ": ";

        #pragma omp parallel
        {
            #pragma omp single
            {
                DFS_recursive(start, visited);
            }
        }

        cout << endl;
    }
};

int main() {
    int V = 8;
    Graph g(V);

    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 3);
    g.addEdge(1, 4);
    g.addEdge(2, 5);
    g.addEdge(2, 6);
    g.addEdge(4, 7);
    g.addEdge(5, 7);

    int startNode = 0;

    auto start_seq = high_resolution_clock::now();
    g.sequentialDFS(startNode);
    auto end_seq = high_resolution_clock::now();
    auto duration_seq = duration_cast<microseconds>(end_seq - start_seq);

    auto start_par = high_resolution_clock::now();
    g.parallelDFS(startNode);
    auto end_par = high_resolution_clock::now();
    auto duration_par = duration_cast<microseconds>(end_par - start_par);

    cout << "\nTime taken by Sequential DFS: " << duration_seq.count() << " microseconds" << endl;
    cout << "Time taken by Parallel DFS: " << duration_par.count() << " microseconds" << endl;

    return 0;
}
