"""Use local ephemeral ports only; verify restart probes against real TCP state."""
import socket
import unittest

from lib.node import free_ports


class StartupPortTests(unittest.TestCase):
    def test_live_listener_blocks_but_closed_connection_does_not(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.settimeout(2)
            listener.bind(('0.0.0.0', 0))
            port = listener.getsockname()[1]
            listener.listen()
            with self.assertRaisesRegex(RuntimeError, 'occupied'):
                free_ports([port])
            with socket.create_connection(('127.0.0.1', port), timeout=2) as client:
                accepted, _ = listener.accept()
                with accepted:
                    # The server closes first, leaving its port in TIME_WAIT.
                    accepted.shutdown(socket.SHUT_WR)
                    self.assertEqual(client.recv(1), b'')
        free_ports([port])


if __name__ == '__main__':
    unittest.main()
