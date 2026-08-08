#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <queue>
#include <algorithm>
#include <atomic>
#include <thread>
#include <optional>
#include <chrono>
#include <cmath>
#include <mutex>
#include <unordered_map>

#include <omp.h>
#include <tbb/concurrent_unordered_map.h>

using State = std::string;

struct AStarNode {
    State state;
    int pushes;
    int priority;
    bool operator>(const AStarNode& other) const { return priority > other.priority; }
};

class Game {
public:
    int width, height;
    std::string board;
    std::string initial_state, goal_boxes;
    std::vector<bool> dead_squares;
    std::vector<std::vector<int>> true_distances;

    void parse_from_file(const std::string& filename) {
        std::ifstream file(filename); std::string line;
        height = 0; width = 0;
        std::string full_map_str;
        while (std::getline(file, line)) {
            if (width == 0) width = line.length();
            full_map_str += line;
            height++;
        }
        uint8_t player_pos_temp;
        std::vector<uint8_t> box_pos_temp, target_pos_temp;
        board.assign(full_map_str.size(), ' ');
        for (int i = 0; i < full_map_str.length(); ++i) {
            char c = full_map_str[i];
            switch(c) {
                case '#': board[i] = '#'; break;
                case 'o': player_pos_temp = i; board[i] = ' '; break;
                case 'O': player_pos_temp = i; board[i] = '.'; target_pos_temp.push_back(i); break;
                case 'x': box_pos_temp.push_back(i); board[i] = ' '; break;
                case 'X': box_pos_temp.push_back(i); board[i] = '.'; target_pos_temp.push_back(i); break;
                case '.': board[i] = '.'; target_pos_temp.push_back(i); break;
                case ' ': board[i] = ' '; break;
                case '@': board[i] = '@'; break;
                case '!': player_pos_temp = i; board[i] = '@'; break;
            }
        }
        std::sort(box_pos_temp.begin(), box_pos_temp.end());
        initial_state += static_cast<char>(player_pos_temp);
        initial_state.append(box_pos_temp.begin(), box_pos_temp.end());
        std::sort(target_pos_temp.begin(), target_pos_temp.end());
        goal_boxes.assign(target_pos_temp.begin(), target_pos_temp.end());
        precompute_dead_squares(); 
        precompute_true_distances();
    }
    bool is_goal(const State& s) const { return s.substr(1) == goal_boxes; }
private:
    void precompute_true_distances() {
        int num_targets = goal_boxes.length();
        int num_squares = width * height;
        true_distances.assign(num_targets, std::vector<int>(num_squares, 1e9));
        const int dr[] = {-1, 1, 0, 0}, dc[] = {0, 0, -1, 1};
        for (int i = 0; i < num_targets; ++i) {
            uint8_t target_pos = goal_boxes[i];
            std::queue<std::pair<int, int>> q;
            q.push({target_pos, 0});
            true_distances[i][target_pos] = 0;
            while (!q.empty()) {
                auto [pos, dist] = q.front(); q.pop();
                for (int j = 0; j < 4; ++j) {
                    int next_pos = pos + dr[j] * width + dc[j];
                    if (board[next_pos] != '#' && true_distances[i][next_pos] > dist + 1) {
                        true_distances[i][next_pos] = dist + 1;
                        q.push({next_pos, dist + 1});
                    }
                }
            }
        }
    }
    void precompute_dead_squares() {
        dead_squares.assign(width * height, true); std::queue<int> q;
        const int dr[] = {-1, 1, 0, 0}, dc[] = {0, 0, -1, 1};
        for (char c : goal_boxes) {
            uint8_t target_pos = c;
            if (dead_squares[target_pos]) { dead_squares[target_pos] = false; q.push(target_pos); }
        }
        while (!q.empty()) {
            int s_pos = q.front(); q.pop();
            for (int i = 0; i < 4; ++i) {
                int p_pos = s_pos - (dr[i] * width + dc[i]);
                int q_pos = p_pos - (dr[i] * width + dc[i]);
                if (p_pos < 0 || q_pos < 0 || p_pos >= width*height || q_pos >= width*height) continue;
                if (board[p_pos] == '#' || board[q_pos] == '#') continue;
                if (dead_squares[p_pos]) { dead_squares[p_pos] = false; q.push(p_pos); }
            }
        }
    }
};

class Solver {
public:
    Solver(Game& game_instance) : game(game_instance) {}

    std::string solve() {
        std::priority_queue<AStarNode, std::vector<AStarNode>, std::greater<AStarNode>> frontier;
        
        std::unordered_map<std::string, int> cost_map;
        tbb::concurrent_unordered_map<State, std::pair<State, std::string>> parent_map;

        AStarNode start_node = {game.initial_state, 0, calculate_heuristic(game.initial_state)};
        frontier.push(start_node);
        parent_map.insert({start_node.state, {start_node.state, ""}});
        cost_map[get_cost_key(start_node.state)] = 0;

        std::atomic<bool> solution_found = false;
        std::optional<State> goal_state;

        const size_t BATCH_SIZE = 256;

        while (!solution_found) {
            std::vector<AStarNode> current_batch;
            current_batch.reserve(BATCH_SIZE);
            
            std::unique_lock<std::mutex> lock(frontier_mutex);
            if (frontier.empty()) break;
            for (size_t i = 0; i < BATCH_SIZE && !frontier.empty(); ++i) {
                current_batch.push_back(std::move(const_cast<AStarNode&>(frontier.top())));
                frontier.pop();
            }
            lock.unlock();
            
            std::vector<std::vector<std::tuple<State, State, std::string, int, int, std::string>>> thread_results(omp_get_max_threads());

            #pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < current_batch.size(); ++i) {
                if (solution_found) continue;

                const AStarNode& current_node = current_batch[i];
                int thread_id = omp_get_thread_num();

                if (game.is_goal(current_node.state)) {
                    if (!solution_found.exchange(true)) goal_state = current_node.state;
                    continue;
                }
                
                if (cost_map.count(get_cost_key(current_node.state)) && cost_map.at(get_cost_key(current_node.state)) < current_node.pushes) {
                    continue;
                }
                
                generate_successors(current_node, thread_results[thread_id]);
            }

            if (solution_found) break;

            lock.lock();
            for (const auto& result_vec : thread_results) {
                for (const auto& item : result_vec) {
                    const auto& next_state = std::get<0>(item);
                    const auto& parent_state = std::get<1>(item);
                    const auto& move_str = std::get<2>(item);
                    const auto& new_cost = std::get<3>(item);
                    const auto& priority = std::get<4>(item);
                    const auto& cost_key = std::get<5>(item); // Use pre-computed key
                    
                    auto it = cost_map.find(cost_key);
                    if (it == cost_map.end() || new_cost < it->second) {
                        cost_map[cost_key] = new_cost;
                        parent_map.insert({next_state, {parent_state, move_str}});
                        frontier.push({next_state, new_cost, priority});
                    }
                }
            }
            lock.unlock();
        }
        
        if (goal_state) return reconstruct_path(*goal_state, parent_map);
        return "No solution found";
    }

private:
    Game& game;
    std::mutex frontier_mutex;
    const int dr[4] = {-1, 1, 0, 0};
    const int dc[4] = {0, 0, -1, 1};
    const char move_chars[4] = {'W', 'S', 'A', 'D'};
    
    std::string get_cost_key(const State& s) {
        std::string key;
        key.reserve(2 + s.length() - 1);
        uint8_t player_pos = s[0];
        
        std::vector<bool> temp_walls(game.width * game.height, false);
        for(size_t i = 1; i < s.length(); ++i) temp_walls[static_cast<uint8_t>(s[i])] = true;

        uint8_t reach_id = player_pos;
        std::queue<int> q;
        q.push(player_pos);
        std::vector<bool> visited(game.width * game.height, false);
        visited[player_pos] = true;

        while(!q.empty()){
            int pos = q.front(); q.pop();
            if(pos < reach_id) reach_id = pos;
            for (int i = 0; i < 4; ++i) {
                int next_pos = pos + dr[i] * game.width + dc[i];
                if (game.board[next_pos] != '#' && !temp_walls[next_pos] && !visited[next_pos]) {
                    visited[next_pos] = true;
                    q.push(next_pos);
                }
            }
        }
        key += static_cast<char>(reach_id);
        key += '|'; 
        key.append(s.data() + 1, s.length() - 1);
        return key;
    }

    int calculate_heuristic(const State& s) {
        std::string boxes = s.substr(1);
        int num_boxes = boxes.length();
        int total_dist = 0;
        std::vector<bool> box_assigned(num_boxes, false);
        std::vector<bool> target_assigned(num_boxes, false);
        for (int k = 0; k < num_boxes; ++k) {
            int best_dist = 1e9;
            int best_box_idx = -1, best_target_idx = -1;
            for (int i = 0; i < num_boxes; ++i) {
                if (box_assigned[i]) continue;
                for (int j = 0; j < num_boxes; ++j) {
                    if (target_assigned[j]) continue;
                    int dist = game.true_distances[j][static_cast<uint8_t>(boxes[i])];
                    if (dist < best_dist) {
                        best_dist = dist; best_box_idx = i; best_target_idx = j;
                    }
                }
            }
            if (best_box_idx != -1) {
                if (best_dist >= 1e9) return 1e9;
                total_dist += best_dist;
                box_assigned[best_box_idx] = true;
                target_assigned[best_target_idx] = true;
            } else { return 1e9; }
        }
        return total_dist;
    }
    
    void generate_successors(const AStarNode& current_node, 
                             std::vector<std::tuple<State, State, std::string, int, int, std::string>>& local_buffer) {
        uint8_t player_pos = current_node.state[0];
        const char* current_boxes_ptr = current_node.state.data() + 1;
        size_t num_boxes = current_node.state.length() - 1;
        std::vector<int> player_parent_map(game.width * game.height, -1);
        std::queue<int> q;
        q.push(player_pos); player_parent_map[player_pos] = player_pos;
        std::vector<bool> temp_walls(game.width * game.height, false);
        for (size_t i = 0; i < num_boxes; ++i) temp_walls[static_cast<uint8_t>(current_boxes_ptr[i])] = true;
        while (!q.empty()) {
            int pos = q.front(); q.pop();
            for (int i = 0; i < 4; ++i) {
                int next_pos = pos + dr[i] * game.width + dc[i];
                if (game.board[next_pos] != '#' && !temp_walls[next_pos] && player_parent_map[next_pos] == -1) {
                    player_parent_map[next_pos] = pos; q.push(next_pos);
                }
            }
        }
        for (size_t box_idx = 0; box_idx < num_boxes; ++box_idx) {
            uint8_t p = current_boxes_ptr[box_idx];
            for (int move_dir = 0; move_dir < 4; ++move_dir) {
                int new_p = p + dr[move_dir] * game.width + dc[move_dir];
                int push_from = p - (dr[move_dir] * game.width + dc[move_dir]);
                if (game.board[new_p]=='#' || temp_walls[new_p] || game.board[new_p]=='@' || game.dead_squares[new_p] || game.board[push_from]=='#') continue;
                if (player_parent_map[push_from] != -1) {
                    std::string path_str;
                    int curr = push_from;
                    while (curr != player_pos) {
                        int prev = player_parent_map[curr]; int diff = curr - prev;
                        if(diff == -game.width) path_str += 'W'; else if(diff == game.width) path_str += 'S';
                        else if(diff == -1) path_str += 'A'; else if(diff == 1) path_str += 'D';
                        curr = prev;
                    }
                    std::reverse(path_str.begin(), path_str.end());
                    
                    std::string next_state_key;
                    next_state_key.reserve(1 + num_boxes);
                    next_state_key += static_cast<char>(p);
                    std::string next_boxes_sorted;
                    next_boxes_sorted.reserve(num_boxes);
                    bool new_pos_added = false;
                    for (size_t i = 0; i < num_boxes; ++i) {
                        if (i == box_idx) continue;
                        if (!new_pos_added && static_cast<uint8_t>(current_boxes_ptr[i]) > new_p) {
                            next_boxes_sorted += static_cast<char>(new_p); new_pos_added = true;
                        }
                        next_boxes_sorted += current_boxes_ptr[i];
                    }
                    if (!new_pos_added) next_boxes_sorted += static_cast<char>(new_p);
                    next_state_key += next_boxes_sorted;
                    
                    int new_cost = current_node.pushes + 1;
                    int priority = new_cost + calculate_heuristic(next_state_key);
                    std::string move_str = path_str + move_chars[move_dir];
                    std::string cost_key = get_cost_key(next_state_key);
                    
                    local_buffer.emplace_back(next_state_key, current_node.state, move_str, new_cost, priority, cost_key);
                }
            }
        }
    }

    std::string reconstruct_path(const State& goal, tbb::concurrent_unordered_map<State, std::pair<State, std::string>>& p_map) {
        std::vector<std::string> path_segments;
        State curr = goal;
        while (curr != game.initial_state) {
            auto it = p_map.find(curr);
            if (it == p_map.end()) return "RECONSTRUCTION_ERROR"; 
            path_segments.push_back(it->second.second);
            curr = it->second.first;
        }
        std::reverse(path_segments.begin(), path_segments.end());
        std::string final_path;
        for (const auto& segment : path_segments) final_path += segment;
        return final_path;
    }
};

int main(int argc, char* argv[]) {
    std::ios_base::sync_with_stdio(false); std::cin.tie(NULL);
    if (argc != 2) { std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl; return 1; }
    Game game; game.parse_from_file(argv[1]);
    Solver solver(game);
    auto start_time = std::chrono::high_resolution_clock::now();
    std::string solution = solver.solve();
    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << solution << std::endl;
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    std::cerr << "Solver finished in: " << elapsed_seconds.count() << " seconds." << std::endl;
    return 0;
}