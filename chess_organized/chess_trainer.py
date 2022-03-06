import io
import pstats
import cProfile
from chessrl import ChessRL


if __name__ == "__main__":
    with cProfile.Profile() as pr:
        C = ChessRL(saved_q='dqn_py_chess', experience_replay_size_threshold=30000)
        C.q_learning(episodes=100000, epsilon=0.5, alpha=0.1, gamma=0.95, max_episode_length=200, show_interval=500)
        # C.sarsa(episodes=10000, epsilon=0.5, alpha=0.1, gamma=0.95, max_episode_length=200, show_interval=500)
        # C.play()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats()

    with open('profile.txt', 'w+') as f:
        f.write(s.getvalue())
