//
// Created by caochao on 7/12/19.
//

#include "../../include/tsp_solver/tsp_solver.h"

// 定义 tsp_solver_ns 命名空间
namespace tsp_solver_ns {
// 构造函数，初始化数据模型
TSPSolver::TSPSolver(tsp_solver_ns::DataModel data)
    : data_(std::move(data)),
      manager_(data_.distance_matrix.size(), data_.num_vehicles, data_.depot),
      routing_(manager_) {}

// 求解 TSP 问题
void TSPSolver::Solve() {
  // 注册过渡回调函数
  const int transit_callback_index = routing_.RegisterTransitCallback(
      [this](int64_t from_index, int64_t to_index) -> int64_t {
        // 将路由变量索引转换为距离矩阵节点索引
        auto from_node = manager_.IndexToNode(from_index).value();
        auto to_node = manager_.IndexToNode(to_index).value();
        return data_.distance_matrix[from_node][to_node];
      });

  // 定义每条弧的成本
  routing_.SetArcCostEvaluatorOfAllVehicles(transit_callback_index);

  // 设置初始解的启发式方法
  RoutingSearchParameters searchParameters = DefaultRoutingSearchParameters();
  searchParameters.set_first_solution_strategy(
      FirstSolutionStrategy::PATH_CHEAPEST_ARC);

  // 求解问题
  solution_ = routing_.SolveWithParameters(searchParameters);
}

void TSPSolver::PrintSolution() {
  // Inspect solution.
  std::cout << "Objective: " << (solution_->ObjectiveValue()) / 10.0
            << " meters" << std::endl;
  int64_t index = routing_.Start(0);
  std::cout << "Route:";
  int64_t distance{0};
  std::stringstream route;
  while (routing_.IsEnd(index) == false) {
    route << manager_.IndexToNode(index).value() << " -> ";
    int64_t previous_index = index;
    index = solution_->Value(routing_.NextVar(index));
    distance += const_cast<RoutingModel &>(routing_).GetArcCostForVehicle(
        previous_index, index, 0LL);
  }
  std::cout << route.str() << manager_.IndexToNode(index).value();
  std::cout << "Route distance: " << distance / 10.0 << " meters";
  std::cout << "Problem solved in " << routing_.solver()->wall_time() << "ms";
}

int TSPSolver::getComputationTime() { return routing_.solver()->wall_time(); }

void TSPSolver::getSolutionNodeIndex(std::vector<int> &node_index,
                                     bool has_dummy) {
  node_index.clear();
  int64_t index = routing_.Start(0);
  int64_t end_index = index;
  while (routing_.IsEnd(index) == false) {
    node_index.push_back(static_cast<int>(manager_.IndexToNode(index).value()));
    index = solution_->Value(routing_.NextVar(index));
  }
  // push back the end node index
  //       node_index.push_back(end_index);
  if (has_dummy) {
    int dummy_node_index = data_.distance_matrix.size() - 1;
    if (node_index[1] == dummy_node_index) {
      // delete dummy node
      node_index.erase(node_index.begin() + 1);
      // push the start node to the end
      node_index.push_back(node_index[0]);
      // remove the start node at the begining
      node_index.erase(node_index.begin());
      // reverse the whole array
      std::reverse(node_index.begin(), node_index.end());
    } else // the last node is dummy node
    {
      node_index.pop_back();
    }
  }
}

double TSPSolver::getPathLength() {
  return (solution_->ObjectiveValue()) / 10.0;
}

} // namespace tsp_solver_ns
