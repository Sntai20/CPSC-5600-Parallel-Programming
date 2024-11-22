#include "cluster.h"
#include <iostream>
#include <limits>
#include <mpi.h>




std::string const USAGE = "Program requires exactly two arguments, both positive integers.\n";

// desc: Reports incorrect command-line usage and exits with status 1
//  pre: None
// post: In description
void bad_usage() {
    std::cerr << USAGE;
    std::exit(1);
}

// desc: Returns the value and thread counts provided by the supplied
//       command-line arguments.
//  pre: There should be exactly two arguments, both positive integers.
//       If this precondition is not met, the program will exit with
//       status 1
// post: In description
void get_args(int argc, char *argv[], int &cluster_count, int &point_count) {

    if (argc <= 2) {
        bad_usage();
    } else if (argc >= 4) {
        bad_usage();
    }

    cluster_count = 0;
    point_count   = 0;

    cluster_count = atoi(argv[1]);
    point_count   = atoi(argv[2]);

    if ((cluster_count <= 0) || (point_count <= 0)) {
        bad_usage();
    }
}



// desc: Returns the distance between the input points
float distance(Point const& a, Point const& b) {
    float x_diff = a.x-b.x;
    float y_diff = a.y-b.y;
    return sqrt(x_diff*x_diff+y_diff*y_diff);
}



// desc: Redefines each cluster center as the average of its cluster's points
bool recenter(ClusterList cluster_list, std::vector<Point>& centers) {
    size_t cluster_count = cluster_list.size();
    bool converged = true;
    for(size_t i=0; i<cluster_count; i++){
        if(cluster_list[i].empty()) {
            continue;
        }
        Point avg {0,0};
        for(Point& point : cluster_list[i]) {
            avg.x += point.x;
            avg.y += point.y;
        }
        size_t size = cluster_list[i].size();
        avg.x /= size;
        avg.y /= size;
        bool x_matches = (centers[i].x == avg.x);
        bool y_matches = (centers[i].y == avg.y);
        if( !x_matches || !y_matches ) {
            converged = false;
        }
        centers[i] = avg;
    }
    return converged;
}


// desc: Re-assign each point to the cluster with the closest center point
ClusterList reassign(ClusterList cluster_list, std::vector<Point> centers) {
    ClusterList result;
    size_t cluster_count = cluster_list.size();
    result.resize(cluster_count);
    for(Cluster& cluster : cluster_list) {
        for(Point& point : cluster) {
            size_t best = 0;
            float best_distance = std::numeric_limits<float>::infinity();
            for(size_t i=0; i<cluster_count; i++){
                float dist = distance(point,centers[i]);
                if(dist < best_distance) {
                    best = i;
                    best_distance = dist;
                }
            }
            result[best].push_back(point);
        }
    }
    return result;
}


// desc: A serial version of k_means
ClusterList k_means(std::vector<Point> points,size_t cluster_count) {
    std::vector<Point> centers;
    centers.resize(cluster_count);

    // Give each center a random offset within our bounds
    // (for this project, we assume both x and y ar between 0 and 1)
    for(Point& point : centers) {
        point = {(float)(rand()%1000)/1000,(float)(rand()%1000)/1000};
    }

    ClusterList clusters;
    clusters.resize(cluster_count);
    clusters[0] = points;
    bool converged = false;
    size_t iteration_limit = 50;
    size_t iteration_count = 0;
    while( (!converged) && (iteration_count < iteration_limit) ) {
        clusters  = reassign(clusters,centers);
        converged = recenter(clusters,centers);
        iteration_count++;
    }
    return clusters;
}


// desc: This is the function that you need to parallelize with MPI
//       The signature of this function should not be changed.
ClusterList parallel_k_means(std::vector<Point> points, size_t cluster_count) {
    
    // Get the rank of the current process.
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Get the total number of processes.
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialize the cluster centers on the root process.
    std::vector<Point> centers(cluster_count);
    if (rank == 0) {
        srand(time(0));
        for (Point& point : centers) {
            point = {(float)(rand() % 1000) / 1000, (float)(rand() % 1000) / 1000};
        }
    }

    // Broadcast the cluster centers.
    MPI_Bcast(
        centers.data(),
        cluster_count * sizeof(Point),
        MPI_BYTE,
        0,
        MPI_COMM_WORLD);

    // Scatter the points to all processes.
    size_t points_per_process = points.size() / size;
    std::vector<Point> local_points(points_per_process);
    MPI_Scatter(
        points.data(),
        points_per_process * sizeof(Point),
        MPI_BYTE,
        local_points.data(),
        points_per_process * sizeof(Point),
        MPI_BYTE,
        0,
        MPI_COMM_WORLD);

    // Perform the k-means algorithm.
    ClusterList clusters(cluster_count);
    bool converged = false;
    size_t iteration_limit = 50;
    size_t iteration_count = 0;

    // Initialize the clusters with the local points.
    while ((!converged) && (iteration_count < iteration_limit)) {
        ClusterList local_clusters = reassign(clusters, centers);

        // Reduce the local clusters to the global clusters.
        std::vector<Point> new_centers(cluster_count);
        std::vector<int> counts(cluster_count, 0);

        // Calculate the new centers.
        for (size_t i = 0; i < cluster_count; ++i) {

            // Initialize the new centers.
            for (const Point& point : local_clusters[i]) {
                new_centers[i].x += point.x;
                new_centers[i].y += point.y;
                counts[i]++;
            }
        }

        // Reduce the new centers.
        MPI_Allreduce(
            MPI_IN_PLACE,
            new_centers.data(),
            cluster_count * sizeof(Point),
            MPI_BYTE,
            MPI_SUM,
            MPI_COMM_WORLD);

        // Reduce the counts.
        MPI_Allreduce(
            MPI_IN_PLACE,
            counts.data(),
            cluster_count,
            MPI_INT,
            MPI_SUM,
            MPI_COMM_WORLD);

        // Calculate the new centers.
        for (size_t i = 0; i < cluster_count; ++i) {
            if (counts[i] > 0) {
                new_centers[i].x /= counts[i];
                new_centers[i].y /= counts[i];
            }
        }

        // Check for convergence.
        if (rank == 0) {
            converged = true;
            for (size_t i = 0; i < cluster_count; ++i) {
                if (distance(new_centers[i], centers[i]) > 1e-4) {
                    converged = false;
                    centers[i] = new_centers[i];
                }
            }
        }

        // Broadcast the convergence status.
        MPI_Bcast(
            &converged,
            1,
            MPI_C_BOOL,
            0,
            MPI_COMM_WORLD);

        // Broadcast the new centers.
        MPI_Bcast(
            centers.data(),
            cluster_count * sizeof(Point),
            MPI_BYTE,
            0,
            MPI_COMM_WORLD);

        iteration_count++;
    }

    // Gather the local points to the global points.
    MPI_Gather(
        local_points.data(),
        points_per_process * sizeof(Point),
        MPI_BYTE,
        points.data(),
        points_per_process * sizeof(Point),
        MPI_BYTE,
        0,
        MPI_COMM_WORLD);

    return clusters;
}


int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int cluster_count, point_count;
    get_args(argc,argv,cluster_count,point_count);

    srand(time(nullptr));
    Point lower_bounds = {0,0};
    Point upper_bounds = {1,1};

    // Procedurally generate a set of clusters
    ClusterList list = generate_cluster_list(lower_bounds,upper_bounds,cluster_count,point_count);

    // Show how they were generated
    display_clusters(list,{0,0},{1,1},40,true);

    // Make a list of the points, with cluster information removed
    std::vector<Point> collapsed = collapse_cluster_list(list);

    // Attempt to find the clusters again
    // ClusterList kmeans = k_means(collapsed,cluster_count);
    ClusterList kmeans = parallel_k_means(collapsed, cluster_count);

    // Display the clusters, as found by the k_means algorithms
    display_clusters(kmeans,{0,0},{1,1},40,true);

    MPI_Finalize();
    return 0;
}