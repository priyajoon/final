#include <iostream>
#include <vector>
#include <queue>
#include <chrono> // For time measurement
#include <omp.h>  // OpenMP header

using namespace std;
using namespace chrono; // For easier time handling

// Graph class using adjacency list representation
class Graph {
private:
    int V; // Number of vertices
    vector<vector<int>> adj; // Adjacency list

public:
    // Constructor
    Graph(int vertices) {
        V = vertices;
        adj.resize(V);
    }

    // Add an undirected edge
    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // Sequential BFS implementation
    void sequentialBFS(int start) {
        vector<bool> visited(V, false); // Track visited nodes
        queue<int> q;

        visited[start] = true;
        q.push(start);

        cout << "Sequential BFS Traversal starting from node " << start << ": ";

        while (!q.empty()) {
            int current = q.front();
            q.pop();
            cout << current << " ";

            for (int neighbor : adj[current]) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    q.push(neighbor);
                }
            }
        }
        cout << endl;
    }

    // Parallel BFS implementation using OpenMP
    void parallelBFS(int start) {
        vector<bool> visited(V, false);
        queue<int> q;

        visited[start] = true;
        q.push(start);

        cout << "Parallel BFS Traversal starting from node " << start << ": ";

        while (!q.empty()) {
            int size = q.size();

            // Process all nodes at the current level in parallel
            #pragma omp parallel for
            for (int i = 0; i < size; i++) {
                int current;

                // Synchronize queue operations
                #pragma omp critical
                {
                    if (!q.empty()) {
                        current = q.front();
                        q.pop();
                        cout << current << " ";
                    }
                }

                // Visit all neighbors
                for (int neighbor : adj[current]) {
                    if (!visited[neighbor]) {
                        #pragma omp critical
                        {
                            if (!visited[neighbor]) {
                                visited[neighbor] = true;
                                q.push(neighbor);
                            }
                        }
                    }
                }
            }
        }

        cout << endl;
    }
};

int main() {
    int V = 8; // Number of vertices
    Graph g(V);

    // Creating a sample undirected graph
    int v;
    cout<<"Enter number of vertices = ";
    cin>>v;
    int n;
    cout<<"Enter number of edges = ";
    cin>>n;
    for(int i=0;i<n;i++){
        int x,y;
        cout<<"Enter the current edge nodes named(0 to n-1) : ";

        cin>>x>>y;
        g.addEdge(x,y);
    }

    // g.addEdge(0, 1);
    // g.addEdge(0, 2);
    // g.addEdge(1, 3);
    // g.addEdge(1, 4);
    // g.addEdge(2, 5);
    // g.addEdge(2, 6);
    // g.addEdge(4, 7);
    // g.addEdge(5, 7);

    int startNode = 0;

    // Measure time for Sequential BFS
    auto start_seq = high_resolution_clock::now();
    g.sequentialBFS(startNode);
    auto stop_seq = high_resolution_clock::now();
    auto duration_seq = duration_cast<microseconds>(stop_seq - start_seq);

    // Measure time for Parallel BFS
    auto start_par = high_resolution_clock::now();
    g.parallelBFS(startNode);
    auto stop_par = high_resolution_clock::now();
    auto duration_par = duration_cast<microseconds>(stop_par - start_par);

    // Print timing results
    cout << "\nTime taken by Sequential BFS: " << duration_seq.count() << " microseconds" << endl;
    cout << "Time taken by Parallel BFS: " << duration_par.count() << " microseconds" << endl;

    return 0;
}
