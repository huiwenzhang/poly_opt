#include "poly_traj_generation/io.h"

#include <boost/filesystem.hpp>
#include <fstream>

namespace poly_opt {
bool saveVerticesToFile(const std::string& file,
                        const Vertex::Vector& vertices) {
    // Eigen::IOFormat csv_format(3, 0, " ", "\n");
    std::cerr << file << std::endl;

    namespace bfs = boost::filesystem;
    bfs::path path(file);

    if (!bfs::exists(path)) {
        bfs::create_directory(path);
        std::cout << "Create file: " << file << std::endl;
    }

    std::ofstream out(file);
    if (!out.is_open()) {
        std::cerr << "failed to open file: " << file << std::endl;
        return false;
    }
    const int D = vertices.front().D();

    out << "DIMENSION: " << D << "\n";
    for (auto& vertex : vertices) {
        Eigen::VectorXd pos(vertex.D());
        if (vertex.getConstraint(derivative_order::POSITION, &pos)) {
            // out << pos.transpose().format(csv_format) << "\n";
            for (int i = 0; i < D - 1; i++) { out << pos(i) << " "; }
            out << pos(D - 1) << "\n";
        }
    }
    out.close();
    return true;
}
bool saveTrajectoryToFile(const std::string& filename,
                          const Trajectory& traj,
                          int derivative) {
    namespace bfs = boost::filesystem;

    std::vector<Eigen::VectorXd> results;
    std::vector<double> times;
    traj.evaluateRange(
            0, traj.getMaxTime(), 0.02, derivative, &results, &times);
    const int D = results.front().size();

    assert(results.size() == times.size());
    std::string file =
            filename + positionDerivativeToString(derivative) + ".txt";

    std::cout << file << std::endl;
    bfs::path path;
    if (file[0] == '.') {
        path = bfs::system_complete(file);
        file = path.c_str();
    } else {
        path = bfs::path(file);
    }
    if (!bfs::exists(path)) {
        bfs::create_directory(path);
        std::cout << "Create file: " << file << std::endl;
    }
    std::ofstream out(file);
    if (!out.is_open()) return false;

    out << "DIMENSION: " << D + 1 << "\n";
    int ti = 0;
    for (auto& res : results) {
        out << times[ti] << " ";
        for (int i = 0; i < D - 1; i++) { out << res(i) << " "; }
        out << res(D - 1) << "\n";
        ti += 1;
    }
    out.close();
    return true;
}

bool saveTrajectoryPVA(const std::string& filename, const Trajectory& traj) {
    std::vector<Eigen::VectorXd> pos, vel, acc;
    std::vector<double> times;
    traj.evaluateRange(0, traj.getMaxTime(), 0.02, 0, &pos, &times);
    traj.evaluateRange(0, traj.getMaxTime(), 0.02, 1, &vel, &times);
    traj.evaluateRange(0, traj.getMaxTime(), 0.02, 2, &acc, &times);
    const int D = pos.front().size();
    const int num = pos.size();

    assert(pos.size() == times.size());
    std::ofstream out(filename);
    if (!out.is_open()) return false;

    out << "t"
        << " "
        << "x"
        << " "
        << "y"
        << " "
        << "z"
        << " "
        << "vx"
        << " "
        << "vy"
        << " "
        << "vz"
        << " "
        << "ax"
        << " "
        << "ay"
        << " "
        << "az"
        << "\n";
    int ti = 0;
    for (int i = 0; i < num; i++) {
        out << times[i] << " " << pos[i].x() << " " << pos[i].y() << " "
            << pos[i].z() << " " << vel[i].x() << " " << vel[i].y() << " "
            << vel[i].z() << " " << acc[i].x() << " " << acc[i].y() << " "
            << acc[i].z() << "\n";
    }
    out.close();
    return true;
}

bool verticesFromFile(const std::string& filename,
                      int derivative_to_opt,
                      Vertex::Vector& vertices) {
    std::ifstream istream(filename);
    if (!istream.is_open()) return false;

    std::string line_str;
    std::stringstream sstr;

    std::getline(istream, line_str);  // get dimension of vertex
    sstr << line_str;
    int dimension;
    std::string field;
    sstr >> field >> dimension;
    sstr.clear();

    while (std::getline(istream, line_str)) {
        Vertex vertex(dimension);
        Eigen::VectorXd constraint(dimension);
        sstr << line_str;
        for (int i = 0; i < dimension; i++) { sstr >> constraint(i); }
        sstr.clear();
        vertex.addConstraint(derivative_order::POSITION, constraint);
        vertices.push_back(vertex);
    }
    istream.close();

    vertices.front().makeStartOrEnd(derivative_to_opt);
    vertices.back().makeStartOrEnd(derivative_to_opt);
    return true;
}

}  // namespace poly_opt