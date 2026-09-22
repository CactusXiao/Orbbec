#include "task_backend_client.hpp"
#include <iostream>
#include <stdexcept>

using namespace sync_app;

int main(int argc, char **argv) {
    if(argc != 2) { return 2; }
    TaskBackendClient client(argv[1]);
    std::vector<TaskBackendTask> tasks;
    std::string error;
    auto require = [&](bool ok) { if(!ok) { throw std::runtime_error(error); } };
    try {
        require(client.getAssignedTask("alice", "alice", tasks, &error));
        require(tasks.size() == 1 && tasks[0].taskName == "first");
        require(client.getAssignedTask("bob", "bob", tasks, &error));
        require(tasks.size() == 1 && tasks[0].taskName == "second");
        TaskEpisodeReservation reservation;
        require(!client.reserveEpisode("capture", "bob", "first", "bob", reservation, &error));
        for(int episode = 1; episode <= 2; ++episode) {
            require(client.reserveEpisode("capture", "alice", "first", "alice", reservation, &error));
            require(reservation.episodeNumber == episode);
            require(client.confirmEpisode(reservation.reservationId, "alice", "first", episode,
                                           "", "", 1.0, 30, reservation.reservationId, "alice", tasks, &error));
            require(tasks.size() == 1 && tasks[0].taskName == (episode == 1 ? "first" : "third"));
        }
        require(client.getAssignedTask("alice", "alice", tasks, &error));
        require(tasks.size() == 1 && tasks[0].taskName == "third");
        require(client.getTasks("alice", tasks, &error));
        require(tasks.size() == 3);
        std::cout << "PASS: native client assignment, exclusivity, confirmation and automatic next task\n";
    }
    catch(const std::exception &exc) {
        std::cerr << exc.what() << '\n';
        return 1;
    }
}
