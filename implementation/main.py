from detection import FaceMonitor
import signal
import sys

def main():
    monitor = FaceMonitor()

    def signal_handler(sig, frame):
        print("Exiting... saving logs and releasing resources.")
        monitor.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        print("Starting face monitoring...")
        monitor.process_stream()
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        monitor.cleanup()

if __name__ == "__main__":
    main()
